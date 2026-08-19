from rasgan.config import TrainConfig


def test_validated_runtime_features_remain_opt_in():
    cfg = TrainConfig()

    assert cfg.mixed is False
    assert cfg.xla is False
    assert cfg.grad_reweight is False
    assert cfg.grad_reweight_deterministic_norms is False
