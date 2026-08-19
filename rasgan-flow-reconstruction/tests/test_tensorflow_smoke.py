import pytest


pytestmark = pytest.mark.tensorflow


def test_rrdb_and_conditional_discriminator_forward():
    tf = pytest.importorskip("tensorflow")

    from rasgan.losses.gan import _make_d_input
    from rasgan.models.discriminator import CondPatchD
    from rasgan.models.generator import SRGAN_g

    x = tf.zeros([1, 3, 16, 16], dtype=tf.float32)
    g = SRGAN_g(sr_scale=2, upsample_mode="resizeconv")
    g.init_build(x)
    y = g(x, training=False)
    assert tuple(y.shape) == (1, 3, 32, 32)

    d_input = _make_d_input(y, x, edge_lambda=1.0, p_drop=0.0, training=False)
    assert tuple(d_input.shape) == (1, 12, 32, 32)

    d = CondPatchD()
    d.init_build(d_input)
    logits, features = d(d_input, return_feats=True)
    assert logits.shape[0] == 1
    assert len(features) == 6
