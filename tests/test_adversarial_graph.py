import pytest


pytestmark = pytest.mark.tensorflow


def test_adversarial_discriminator_loss_runs_under_tf_function():
    """Regression test for graph-mode discriminator conditioning.

    The normal eager forward smoke test does not exercise the same control-flow
    path as adversarial training. This test traces the real WithLoss_D wrapper
    under tf.function so Python-vs-Tensor boolean regressions in _make_d_input
    fail in CI before reaching a training run.
    """
    tf = pytest.importorskip("tensorflow")

    from rasgan.losses.gan import _make_d_input
    from rasgan.losses.spectrum import set_spec_border
    from rasgan.losses.wrappers import WithLoss_D
    from rasgan.models.discriminator import CondPatchD
    from rasgan.models.generator import SRGAN_g

    tf.random.set_seed(7)

    # Mirror the normal training initialization. _crop_logits() uses the
    # spectral border frozen at startup by training.py.
    set_spec_border(4)

    lr = tf.zeros([1, 3, 16, 16], dtype=tf.float32)
    hr = tf.zeros([1, 3, 32, 32], dtype=tf.float32)

    # Build model variables eagerly before tracing the adversarial loss.
    g = SRGAN_g(sr_scale=2, upsample_mode="resizeconv")
    g.init_build(lr)

    d_example = _make_d_input(
        hr, lr, edge_lambda=1.0, p_drop=0.0, training=False
    )
    d = CondPatchD()
    d.init_build(d_example)

    # Disable R1 here: this test targets graph tracing of the normal
    # adversarial discriminator path, not the much heavier R1 branch.
    d_loss_net = WithLoss_D(
        d,
        g,
        r1_gamma=0.0,
        inst_noise_std=0.0,
        reg_every=8,
        edge_lambda=1.0,
        cond_drop_p=0.35,
    )

    @tf.function
    def graph_d_loss(lr_batch, hr_batch):
        return d_loss_net(lr_batch, hr_batch, None, training=True)

    loss = graph_d_loss(lr, hr)

    assert loss.shape.rank == 0
    assert bool(tf.math.is_finite(loss).numpy())
