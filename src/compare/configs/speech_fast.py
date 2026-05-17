from ..adapters import (
    CosineAdapter,
    GMMAdapter,
    KNNAdapter,
    PrototypeAdapter,
    SmallAEAdapter,
)
from ..sweep import sweep

PROVIDER = "speech"
# CHECKPOINT = "speech.ckpt"
# CHECKPOINT = "best_16.ckpt"
CHECKPOINT = "best_16_repro.ckpt"
TRAIN_N = list(range(10, 100, 5))


def make_configs(embedding_dim: int, device: str) -> list:
    return [
        *sweep(GMMAdapter, {
            "train_n": TRAIN_N,
            "n_components": [1, 2, 3],
            "covariance_type": ["diag", "full", "spherical"],
        }),
        # *sweep(KNNAdapter, {
        #     "train_n": TRAIN_N,
        #     "k": list(range(1, 6)),
        # }),
        # *sweep(PrototypeAdapter, {"train_n": TRAIN_N}),
        *sweep(CosineAdapter,    {"train_n": TRAIN_N}),
        # *sweep(SmallAEAdapter, {
        #     "train_n": TRAIN_N,
        #     "latent_dim": [4],
        #     "epochs": [50],
        #     "device": [device],
        #     "input_dim": [embedding_dim],
        # }),
    ]
