from __future__ import annotations

"""Transformer-based generator for 2× super-resolution (TF-only, project-native layers).

Why this version:
- Uses this repo's NCHW-first wrappers (Conv2d / UpSampling2d / UpSampleBilinear2d) so:
  * padding, data_format, and initialization conventions match the rest of the project
  * you can swap upsample modes consistently with other SR generators
- Avoids raw `tf.image.resize(...)` in the model body (upsampling happens via tf_layers upsample blocks)

Design:
- Patch-embed LR (NCHW) into tokens, run a ViT-style encoder stack.
- Decode tokens back to a feature map, then:
  (Hp,Wp) --(resizeconv × patch_size)--> LR (H,W)
  LR --(resizeconv × 2)--> HR (2H,2W)
- Predict a *delta* added on top of bilinear-upsampled LR so the model starts near identity.

Conditioning:
- Optional POD time coefficients `pod_coeffs` are turned into a style vector and used for:
  * a learned style token
  * optional FiLM (gamma/beta) per transformer block
"""

from typing import Optional, Sequence, Tuple

from ..env import tf
from ..tf_layers import Module, Conv2d, UpSampling2d, UpSampleBilinear2d, Elementwise, Lambda, Swish, Concat, Conv1d


class _FiLM(tf.keras.layers.Layer):
    def __init__(self, style_dim: int, embed_dim: int):
        super().__init__()
        self.style_dim = int(style_dim)
        self.embed_dim = int(embed_dim)
        self.g1 = tf.keras.layers.Dense(self.embed_dim, activation=None, kernel_initializer="zeros", bias_initializer="zeros")
        self.g2 = tf.keras.layers.Dense(self.embed_dim, activation=None, kernel_initializer="zeros", bias_initializer="zeros")
        self.b1 = tf.keras.layers.Dense(self.embed_dim, activation=None, kernel_initializer="zeros", bias_initializer="zeros")
        self.b2 = tf.keras.layers.Dense(self.embed_dim, activation=None, kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, s: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # s: [B, style_dim]
        g = self.g2(tf.nn.gelu(self.g1(s)))
        b = self.b2(tf.nn.gelu(self.b1(s)))
        return g[:, None, :], b[:, None, :]


class _TransformerBlock(tf.keras.layers.Layer):
    """Pre-Norm encoder block (LN → MHA → +res → LN → MLP → +res)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads (got dim={dim}, heads={num_heads})")

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=float(dropout),
        )
        hidden = int(dim * float(mlp_ratio))
        self.ff1 = tf.keras.layers.Dense(hidden, activation=None)
        self.ff2 = tf.keras.layers.Dense(dim, activation=None)
        self.drop1 = tf.keras.layers.Dropout(float(dropout))
        self.drop2 = tf.keras.layers.Dropout(float(dropout))
        self.drop3 = tf.keras.layers.Dropout(float(dropout))

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        h = self.ln1(x)
        h = self.attn(h, h, training=training)
        h = self.drop1(h, training=training)
        x = x + h

        h = self.ln2(x)
        h = tf.nn.gelu(self.ff1(h))
        h = self.drop2(h, training=training)
        h = self.ff2(h)
        h = self.drop3(h, training=training)
        return x + h


def _style_vec_from_pod(
    pod_coeffs, k: int, style_dim: int, batch_size, dtype=tf.float32,
    # Optional per-dataset normalization for POD coefficients.
    # If provided, (tim_p - p_mean)/p_std and (tim_v - v_mean)/v_std is used.
    # These should be scalars (or broadcastable to [B,K]).
    p_mean: Optional[tf.Tensor] = None,
    p_std: Optional[tf.Tensor] = None,
    v_mean: Optional[tf.Tensor] = None,
    v_std: Optional[tf.Tensor] = None,
):
    """
    Returns [B, style_dim] always (never None).
    style_dim is expected to be 3*k (p,v,omega slots), but we don't assume.
    """
    if pod_coeffs is None:
        return tf.zeros((batch_size, style_dim), dtype=dtype)

    if tf.is_tensor(pod_coeffs):
        x = tf.cast(pod_coeffs, dtype)
        if x.shape.rank == 1:
            x = x[:, None]
        # If caller prepacked, just pad/trim to style_dim for safety
        x = x[:, :style_dim]
        if x.shape.rank == 2 and x.shape[1] != style_dim:
            pad = style_dim - tf.shape(x)[1]
            x = tf.pad(x, [[0, 0], [0, pad]])
        return x

    if isinstance(pod_coeffs, (tuple, list)) and len(pod_coeffs) >= 3:
        tim_u, tim_v, tim_p = pod_coeffs[0], pod_coeffs[1], pod_coeffs[2]
        tim_v = tf.cast(tim_v, dtype)
        tim_p = tf.cast(tim_p, dtype)

        if (v_mean is not None) and (v_std is not None):
            tim_v = (tim_v - tf.cast(v_mean, dtype)) / tf.cast(v_std, dtype)
        if (p_mean is not None) and (p_std is not None):
            tim_p = (tim_p - tf.cast(p_mean, dtype)) / tf.cast(p_std, dtype)

        if tim_v.shape.rank == 1: tim_v = tim_v[:, None]
        if tim_p.shape.rank == 1: tim_p = tim_p[:, None]
        zeros = tf.zeros_like(tim_p)  # omega slot
        x = tf.concat([tim_p, tim_v, zeros], axis=-1)  # (p, v, omega)
        # Pad/trim to style_dim
        x = x[:, :style_dim]
        pad = style_dim - tf.shape(x)[1]
        x = tf.cond(pad > 0, lambda: tf.pad(x, [[0, 0], [0, pad]]), lambda: x)
        return x

    raise TypeError(f"Unsupported pod_coeffs type/shape: {type(pod_coeffs)}")

class ECALayer(Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        k = int(k_size)
        if k % 2 == 0:  # ECA needs odd kernel
            k += 1
        self.channels = channels
        self.conv = Conv1d(
            out_channels=1, kernel_size=k, stride=1, padding='SAME', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_last', b_init=None # we’ll feed (N, C, 1)
        )

    def forward(self, x):
        # x: (N,C,H,W)
        # global avg pool over H,W -> (N,C)
        y = tf.reduce_mean(x, axis=[2, 3])         # (N,C)
        y = tf.expand_dims(y, axis=-1)             # (N,C,1)  (length=C, channels=1)
        y = self.conv(y)                            # (N,C,1)
        y = tf.sigmoid(y)                          # (N,C,1)
        # reshape to (N,C,1,1) and scale
        N = tf.shape(x)[0]; C = tf.shape(x)[1]
        y = tf.reshape(y, (N, C, 1, 1))
        return x * y

    def call(self, x, training=None):
        return self.forward(x)

class ResidualBlock(Module):
    """
    Dense-style residual block with symmetric kernel sizes [3,5,7,5,3] + ECA.
    The current implementation uses dilation=1 throughout.
    """
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.dim = int(dim)
        self.conv1 = Conv2d(
            out_channels=self.dim, kernel_size=(3, 3), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr1 = Swish()
        self.concat1 = Concat(concat_dim=1)

        self.conv2 = Conv2d(
            out_channels=self.dim, kernel_size=(5, 5), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr2 = Swish()
        self.concat2 = Concat(concat_dim=1)

        self.conv3 = Conv2d(
            out_channels=self.dim, kernel_size=(7, 7), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr3 = Swish()
        self.concat3 = Concat(concat_dim=1)

        self.conv4 = Conv2d(
            out_channels=self.dim, kernel_size=(5, 5), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr4 = Swish()
        self.concat4 = Concat(concat_dim=1)

        self.conv5 = Conv2d(
            out_channels=self.dim, kernel_size=(3, 3), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        # Lightweight channel attention on the residual branch
        self.eca = ECALayer(channels=self.dim, k_size=3)
        # Residual scaling (keeps block stable)
        self.lmd = Lambda(lambda x: 0.25 * x)

    def forward(self, x):
        z = self.conv1(x);  z = self.lr1(z)
        z1 = self.concat1([x, z])          # (N,128,H,W)
        z = self.conv2(z1); z = self.lr2(z)
        z2 = self.concat2([z1, z])         # (N,192,H,W)
        z = self.conv3(z2); z = self.lr3(z)
        z3 = self.concat3([z2, z])         # (N,256,H,W)
        z = self.conv4(z3); z = self.lr4(z)
        z4 = self.concat4([z3, z])         # (N,320,H,W)
        z = self.conv5(z4)                 # (N,64,H,W)

        # ECA attention on residual
        z = self.eca(z)
        z = self.lmd(z)

        return x + z

    def call(self, x, training=None):
        return self.forward(x)

def _decompose_scale(scale: int):
    """Return a list of integer upsample factors whose product == scale.
    Prefers repeated 2× stages, then uses the remaining factor (if any).
    Examples:
      4 -> [2,2]
      8 -> [2,2,2]
      6 -> [2,3]
      3 -> [3]
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale == 1:
        return []

    s = int(scale)
    factors = []
    while s % 2 == 0:
        factors.append(2)
        s //= 2
    if s != 1:
        factors.append(s)
    return factors


class CompositeSR_g(Module):
    """Composite generator (2× SR capable), compatible with the repo's NCHW training pipeline."""

    def __init__(
        self,
        *,
        sr_scale: int = 2,
        in_ch: int = 3,
        out_ch: int = 3,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        coeff_dim: int = 12,
        use_film: bool = True,
    ):
        super().__init__()

        self.g_arch = "composite"
        self.sr_scale = int(sr_scale)
        if self.sr_scale not in (1, 2):
            raise ValueError(f"sr_scale must be 1 or 2, got {self.sr_scale}")
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.coeff_dim = int(coeff_dim)
        self.use_film = bool(use_film)

        # Patch embedding directly on NCHW (project-native Conv2d wrapper).
        self.cond_proj = Conv2d(
            out_channels=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            padding="VALID",
            data_format="channels_first",
            act=None,
        )

        self.residual_block1 = ResidualBlock(dim=self.embed_dim)
        self.residual_block2 = ResidualBlock(dim=self.embed_dim)
        self.lmd = Lambda(lambda x: 0.25 * x)
        self.add = Elementwise(combine_fn=tf.add)

        # Style token MLP
        self._style_in = max(1, self.coeff_dim)
        self.style_fc1 = tf.keras.layers.Dense(self.embed_dim, activation=None)
        self.style_fc2 = tf.keras.layers.Dense(self.embed_dim, activation=None)

        # Transformer encoder stack
        self.blocks = [
            _TransformerBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout)
            for i in range(self.depth)
        ]
        self.film = [_FiLM(self._style_in, self.embed_dim) for i in range(self.depth)]

        # Token->map head (still in NCHW)
        self.head = Conv2d(
            out_channels=self.embed_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="SAME",
            data_format="channels_first",
            act="gelu",
        )

        # (Hp,Wp)->(H,W) using resize+conv (scale=patch_size)
        self.up_to_lr = []
        self.lr_refine = []

        for i, sc in enumerate(_decompose_scale(self.patch_size), start=1):
            self.up_to_lr.append(
                UpSampleBilinear2d(
                    out_channels=self.embed_dim,
                    scale=sc,
                    kernel_size=(3, 3),
                    padding="SAME",
                    data_format="channels_first",
                    act="swish",
                )
            )
            self.lr_refine.append(
                Conv2d(
                    out_channels=self.embed_dim,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding="SAME",
                    data_format="channels_first",
                    act=None,
                )
            )
        if self.sr_scale == 2:
            # 2× upsample to HR using resize+conv, then refine
            self.up2 = UpSampleBilinear2d(
                out_channels=self.embed_dim,
                scale=2,
                kernel_size=(3, 3),
                padding="SAME",
                data_format="channels_first",
                act="swish",
            )
            self.hr_refine = Conv2d(
                out_channels=self.embed_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="SAME",
                data_format="channels_first",
                act=None,
            )
            # Base 2× bilinear upsample for residual formulation
            self.base_up2 = UpSampling2d(scale=2, data_format="channels_first")
        else:
            self.up2 = None
            self.hr_refine = None
            self.base_up2 = None

        # Delta head: initialize to zeros so the initial function is ~identity
        self.out_delta = Conv2d(
            out_channels=self.out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="SAME",
            data_format="channels_first",
            act=None,
#            W_init=tf.keras.initializers.HeNormal(),
#            b_init=None, #tf.keras.initializers.Zeros(),
        )

        # Positional embedding allocated lazily in build() because N depends on LR H,W.
        self.pos: Optional[tf.Variable] = None
        self._N: Optional[int] = None
        self._Hp: Optional[int] = None
        self._Wp: Optional[int] = None

        # ---- POD coefficient normalization (dataset-specific) ----
        # We want conditioning to work across datasets without hard-coded constants.
        # Stored as non-trainable weights so they are saved with checkpoints, but can
        # be overridden at runtime via `set_coeff_norm_from_hr(...)`.
        self.pod_coeff_mean_pv = self.add_weight(
            name="pod_coeff_mean_pv",
            shape=(2,),
            initializer="zeros",
            trainable=False,
        )
        self.pod_coeff_std_pv = self.add_weight(
            name="pod_coeff_std_pv",
            shape=(2,),
            initializer="ones",
            trainable=False,
        )

    def set_coeff_norm_from_hr(self, hr_mean: tf.Tensor, hr_std: tf.Tensor) -> None:
        """Set POD coefficient normalization from dataset HR stats.
        Expects HR channels in (p, v, ω) order. We use p and v.
        hr_mean/hr_std are broadcastable (1,C,1,1) tensors from RuntimeStats.
        """
        try:
            m = tf.reshape(tf.cast(hr_mean, tf.float32), [-1])
            s = tf.reshape(tf.cast(hr_std, tf.float32), [-1])
            # Need at least (p,v)
            if int(m.shape[0]) < 2 or int(s.shape[0]) < 2:
                return
            mean_p = m[0]
            mean_v = m[1]
            std_p = tf.maximum(s[0], tf.constant(1e-6, tf.float32))
            std_v = tf.maximum(s[1], tf.constant(1e-6, tf.float32))
            self.pod_coeff_mean_pv.assign(tf.stack([mean_p, mean_v], axis=0))
            self.pod_coeff_std_pv.assign(tf.stack([std_p, std_v], axis=0))
        except Exception:
            return  # Best-effort: leave defaults (no normalization).

    def build(self, input_shape):
        # input_shape: (B, C, H, W) for NCHW
        if len(input_shape) != 4:
            raise ValueError(f"Expected NCHW input shape rank 4, got {input_shape}")
        H = int(input_shape[2])
        W = int(input_shape[3])
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"LR H,W must be divisible by patch_size={self.patch_size}. Got H={H}, W={W}."
            )
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        N = Hp * Wp
        self._Hp, self._Wp, self._N = int(Hp), int(Wp), int(N)
        if self.pos is None:
            self.pos = self.add_weight(
                shape=(1, N + 1, self.embed_dim),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
            )
        super().build(input_shape)

    def forward(self, lr, pod_coeffs=None, training=True):
        # lr: [B,C,H,W] (NCHW)
        B = tf.shape(lr)[0]

        # 1) Patch embedding (NCHW -> tokens)
        t = self.cond_proj(lr)  # [B,embed,Hp,Wp]
        t1 = t
        Hp = tf.shape(t)[2]
        Wp = tf.shape(t)[3]
        t = self.residual_block1(t)
        t = self.residual_block2(t)
        t = self.lmd(t)
        t = self.add([t1, t])

        # NCHW -> NHWC for tokenization
        t_nhwc = tf.transpose(t, [0, 2, 3, 1])  # [B,Hp,Wp,embed]
        t_tok = tf.reshape(t_nhwc, [B, -1, self.embed_dim])  # [B,N,embed]

        # 2) Style token
        svec = _style_vec_from_pod(
            pod_coeffs,
            k=(self.coeff_dim // 3),
            style_dim=self._style_in,
            batch_size=B,
            dtype=lr.dtype,
            p_mean=self.pod_coeff_mean_pv[0],
            p_std=self.pod_coeff_std_pv[0],
            v_mean=self.pod_coeff_mean_pv[1],
            v_std=self.pod_coeff_std_pv[1],
        )
        s = self.style_fc2(tf.nn.gelu(self.style_fc1(svec)))  # [B,embed]
        s = s[:, None, :]  # [B,1,embed]

        x = tf.concat([s, t_tok], axis=1)
        x = x + self.pos[:, : tf.shape(x)[1], :]

        if self.use_film:
            gammas = []
            betas = []
            for i in range(self.depth):
                g, b = self.film[i](svec)
                gammas.append(g)
                betas.append(b)
        else:
            gammas = betas = [None] * self.depth

        for i, blk in enumerate(self.blocks):
            if self.use_film:
                x = x * (1.0 + gammas[i]) + betas[i]
            x = blk(x, training=training)  # training context flows from outer Keras call()

        # 4) Tokens -> feature map (back to NCHW)
        x = x[:, 1:, :]  # [B,N,embed]
        x = tf.reshape(x, [B, Hp, Wp, self.embed_dim])   # NHWC
        x = tf.transpose(x, [0, 3, 1, 2])               # NCHW

        # 5) Decode: (Hp,Wp)->(H,W)->(2H,2W)
        x = self.head(x)
        for up, ref in zip(self.up_to_lr, self.lr_refine):
            x = up(x)
            x = ref(x)
        if self.up2 is not None:
            x = self.up2(x)
            x = self.hr_refine(x)
        delta = self.out_delta(x)  # [B,out_ch,2H,2W]

        # 6) Residual on top of bilinear-upsampled LR
        base = self.base_up2(lr) if self.sr_scale == 2 else lr # [B,in_ch,2H,2W]
        out = base[:, : self.out_ch, :, :] + delta
        return delta

    def call(self, lr, pod_coeffs=None, training=None):
        return self.forward(lr, pod_coeffs=pod_coeffs, training=training)
