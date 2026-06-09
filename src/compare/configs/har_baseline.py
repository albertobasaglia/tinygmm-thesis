from ..adapters import (
    CosineAdapter,
    GMMAdapter,
    KNNAdapter,
    PrototypeAdapter,
    SmallAEAdapter,
)
from ..sweep import sweep
from .frozen import make_test_configs as _make_test_configs

PROVIDER = "har"
CHECKPOINT = "checkpoints/har/best.ckpt"
TRAIN_N = list(range(5, 200, 5))
 


def make_test_configs(embedding_dim: int, device: str) -> list:
    return _make_test_configs(embedding_dim, device, TRAIN_N)


def make_configs(embedding_dim: int, device: str) -> list:
    return [
        *sweep(GMMAdapter, {
            "train_n": TRAIN_N,
            "n_components": [1, 2, 3],
            "covariance_type": ["diag", "full", "spherical"],
        }),
        *sweep(KNNAdapter, {
            "train_n": TRAIN_N,
            "k": list(range(1, 6)),
        }),
        *sweep(PrototypeAdapter, {"train_n": TRAIN_N}),
        *sweep(CosineAdapter,    {"train_n": TRAIN_N}),
        *sweep(SmallAEAdapter, {
            "train_n": TRAIN_N,
            "latent_dim": [4, 8],
            "epochs": [50, 100],
            "device": [device],
            "input_dim": [embedding_dim],
        }),
    ]
