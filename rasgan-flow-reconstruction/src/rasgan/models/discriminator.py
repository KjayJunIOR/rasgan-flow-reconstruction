from __future__ import annotations
from ..env import tf
from ..tf_layers import (
    Module, Conv2d, GlobalAvgPool2d, Flatten, Linear, BatchNorm2d, MaxPool2d
)

from ..config import D_DIMS

d_dimsize = D_DIMS

def _l2_normalize(v, eps=1e-12):
    return v / (tf.sqrt(tf.reduce_sum(v**2)) + eps)

def _sn_power_iteration(W_mat, u, n_power=1, eps=1e-12):
    u_hat = u
    for _ in range(n_power):
        v_hat = _l2_normalize(tf.matmul(tf.transpose(W_mat), tf.reshape(u_hat, [-1,1]))[:,0], eps)
        u_hat = _l2_normalize(tf.matmul(W_mat, tf.reshape(v_hat, [-1,1]))[:,0], eps)
    sigma = tf.tensordot(u_hat, tf.matmul(W_mat, tf.reshape(v_hat, [-1,1]))[:,0], axes=1)
    return sigma, u_hat

def _sn_conv_kernel(W, u):
    W_mat = tf.reshape(tf.transpose(W, [3,2,0,1]), [tf.shape(W)[3], -1])
    sigma, u_new = _sn_power_iteration(W_mat, u, n_power=1)
    return W/(sigma+1e-12), u_new

def _sn_dense_kernel(W, u):
    W_mat = tf.transpose(W)
    sigma, u_new = _sn_power_iteration(W_mat, u, n_power=1)
    return W/(sigma+1e-12), u_new

class SNConv2d(Module):
    """Conv2D with spectral normalization.

    NOTE: Keras may mark a layer as "built" even when it calls build(None). Once a
    layer is marked built, Keras forbids creating new variables later. To avoid
    that, we allow passing `in_channels` so variables are created up-front.
    """
    def __init__(
        self,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding='SAME',
        act='lrelu',
        data_format='channels_first',
        use_bias=False,
        in_channels=None,
        name=None,
    ):
        super().__init__(name=name)
        self.out_channels = int(out_channels)
        self.kh, self.kw = int(kernel_size[0]), int(kernel_size[1])
        self.sh, self.sw = int(stride[0]), int(stride[1])
        self.padding = padding.upper()
        self.act = act
        self.data_format = 'NCHW' if data_format == 'channels_first' else 'NHWC'
        self.use_bias = bool(use_bias)
        self.in_channels = int(in_channels) if in_channels is not None else None

        # Create variables eagerly if possible.
        if self.in_channels is not None:
            self._create_vars(self.in_channels)

    def _create_vars(self, inC: int):
        if hasattr(self, 'W'):
            return
        self.W = self._get_weights(
            'W',
            (self.kh, self.kw, int(inC), self.out_channels),
            tf.keras.initializers.HeNormal(),
        )
        if self.use_bias:
            self.b = self._get_weights('b', (self.out_channels,), tf.keras.initializers.Zeros())
        u_init = tf.random.normal([self.out_channels])
        u_init = u_init / (tf.sqrt(tf.reduce_sum(u_init**2)) + 1e-12)
        self.u = self._get_weights(
            'u',
            (self.out_channels,),
            tf.keras.initializers.Constant(u_init.numpy()),
            trainable=False,
        )

    def build(self, inputs_shape):
        # If variables were created in __init__, nothing to do.
        if hasattr(self, 'W'):
            return
        if inputs_shape is None:
            return
        inC_raw = (inputs_shape[1] if self.data_format == 'NCHW' else inputs_shape[-1])
        if inC_raw is None:
            return
        self._create_vars(int(inC_raw))

    def forward(self, x):
        # Ensure variables exist. If we reach here without W, inference failed.
        if not hasattr(self, 'W'):
            # Try to infer from runtime tensor shape.
            inC = int(x.shape[1] if self.data_format == 'NCHW' else x.shape[-1])
            if inC is None:
                raise ValueError('SNConv2d could not infer input channels; got shape %r' % (x.shape,))
            # If Keras has already marked the layer built, add_weight would fail;
            # therefore we strongly prefer providing in_channels at init.
            self._create_vars(inC)

        W_bar, u_new = _sn_conv_kernel(self.W, self.u)
        self.u.assign(tf.stop_gradient(u_new))
        strides = [1, 1, self.sh, self.sw] if self.data_format == 'NCHW' else [1, self.sh, self.sw, 1]
        y = tf.nn.conv2d(x, W_bar, strides=strides, padding=self.padding, data_format=self.data_format)
        if self.use_bias:
            y = y + (
                tf.reshape(self.b, (1, -1, 1, 1))
                if self.data_format == 'NCHW'
                else tf.reshape(self.b, (1, 1, 1, -1))
            )
        if self.act == 'lrelu':
            y = tf.nn.leaky_relu(y, alpha=0.2)
        return y

class SNLinear(Module):
    def __init__(self, out_features, act=None, in_features=None, name=None):
        super().__init__(name=name)
        self.out_features = int(out_features)
        self.act = act
        self.in_features = int(in_features) if in_features is not None else None
        if self.in_features is not None:
            self._create_vars(self.in_features)

    def _create_vars(self, inF: int):
        if hasattr(self, 'W'):
            return
        self.W = self._get_weights(
            'W',
            (int(inF), self.out_features),
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.b = self._get_weights('b', (self.out_features,), tf.keras.initializers.Zeros())
        u_init = tf.random.normal([self.out_features])
        u_init = u_init / (tf.sqrt(tf.reduce_sum(u_init**2)) + 1e-12)
        self.u = self._get_weights(
            'u',
            (self.out_features,),
            tf.keras.initializers.Constant(u_init.numpy()),
            trainable=False,
        )

    def build(self, inputs_shape):
        if hasattr(self, 'W'):
            return
        if inputs_shape is None:
            return
        inF_raw = inputs_shape[-1]
        if inF_raw is None:
            return
        self._create_vars(int(inF_raw))

    def forward(self, x):
        if not hasattr(self, 'W'):
            inF = int(x.shape[-1])
            if inF is None:
                raise ValueError('SNLinear could not infer input features; got shape %r' % (x.shape,))
            self._create_vars(inF)
        W_bar, u_new = _sn_dense_kernel(self.W, self.u)
        self.u.assign(tf.stop_gradient(u_new))
        y = tf.matmul(x, W_bar) + self.b
        if self.act == 'lrelu':
            y = tf.nn.leaky_relu(y, alpha=0.2)
        return y

class SEBlock(Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = Linear(out_features=max(1, c // r))
        self.fc2 = Linear(out_features=c)
    def forward(self, x):
        s = tf.reduce_mean(x, axis=[2,3], keepdims=True)
        z = tf.reshape(s, [tf.shape(s)[0], tf.shape(s)[1]])
        z = tf.nn.relu(self.fc1(z))
        z = tf.nn.sigmoid(self.fc2(z))
        z = tf.reshape(z, [tf.shape(s)[0], tf.shape(s)[1], 1, 1])
        return x * z

# Conditional PatchGAN; returns features for FM in Stage B
class CondPatchD(Module):
    def __init__(self, dim=d_dimsize):
        super().__init__()
        Cproj = dim
        self.proj = Conv2d(out_channels=Cproj, kernel_size=(1,1), stride=(1,1),
                           act=None, padding='SAME', W_init=tf.keras.initializers.HeNormal(),
                           data_format='channels_first', b_init=None)
        self.se = SEBlock(Cproj, r=8)
        # Provide in_channels so SN layers can create variables in __init__.
        # This avoids Keras build(None) edge cases that would otherwise prevent
        # variable creation later.
        self.c1 = SNConv2d(dim,   in_channels=dim,     name='c1')
        self.c2 = SNConv2d(dim*2, in_channels=dim,     name='c2')
        self.c3 = SNConv2d(dim*4, in_channels=dim*2,   name='c3')
        self.c4 = SNConv2d(dim*8, in_channels=dim*4,   name='c4')
        self.c5 = Conv2d(out_channels=dim*8, kernel_size=(3,3), stride=(1,1),
                         act=None, padding='SAME', W_init=tf.keras.initializers.HeNormal(),
                         data_format='channels_first', b_init=None, name='c5')
        self.c6 = Conv2d(out_channels=dim*4, kernel_size=(3,3), stride=(1,1),
                         act=None, padding='SAME', W_init=tf.keras.initializers.HeNormal(),
                         data_format='channels_first', b_init=None, name='c6')
        self.gap = GlobalAvgPool2d(data_format='channels_first')
        self.global_fc = SNLinear(1, in_features=dim*4)
        self.head = Conv2d(out_channels=1, kernel_size=(1,1), stride=(1,1), act=None,
                           padding='SAME', W_init=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                           data_format='channels_first', b_init=None, name='head')

    def forward(self, x_cond, return_feats=False):
        feats = []
        x = self.proj(x_cond)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.se(x)
        def _act(m, x):
            y = m(x); return tf.nn.leaky_relu(y, 0.2)
        x = _act(self.c1, x); feats.append(x)
        x = _act(self.c2, x); feats.append(x)
        x = _act(self.c3, x); feats.append(x)
        x = _act(self.c4, x); feats.append(x)
        x = _act(self.c5, x); feats.append(x)
        x = _act(self.c6, x); feats.append(x)
        logits_patch = self.head(x)
        g = self.global_fc(self.gap(x))
        logits = logits_patch + tf.reshape(g, [-1,1,1,1])
        return (logits, feats) if return_feats else logits
