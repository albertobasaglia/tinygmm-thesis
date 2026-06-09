"""Single source of truth for the frozen best-of-each adaptive-layer set.

The main sweep selects configurations on the validation classes; the final test
confirms a *frozen* shortlist on the held-out test classes. That shortlist must
match the "best of each family" used by the plotting/export code, or the test
parquet and the comparison figures could silently drift apart. To prevent that,
both `make_test_configs()` (in each dataset config module) and
`export_plots.BEST_LINES` derive from the single list below.

Each entry is `(label, AdapterClass, params)` where `params` are the
distinguishing hyperparameters of that family (the enrollment budget `train_n`
and the AE's `input_dim`/`device` are injected per dataset/run, not stored
here).
"""

from ..adapters import (
    CosineAdapter,
    GMMAdapter,
    KNNAdapter,
    PrototypeAdapter,
    SmallAEAdapter,
)

# (label, adapter class, distinguishing params)
FROZEN_BEST = [
    ("GMM K=1 full", GMMAdapter,       {"n_components": 1, "covariance_type": "full"}),
    ("GMM K=1 diag", GMMAdapter,       {"n_components": 1, "covariance_type": "diag"}),
    ("kNN k=5",      KNNAdapter,       {"k": 5}),
    ("AE",           SmallAEAdapter,   {"latent_dim": 8, "epochs": 100}),
    ("Cosine",       CosineAdapter,    {}),
    ("Prototype",    PrototypeAdapter, {}),
]


def best_lines() -> list[tuple[str, dict]]:
    """`(label, p_-prefixed filter dict)` pairs for export_plots.BEST_LINES."""
    lines = []
    for label, cls, params in FROZEN_BEST:
        where = {"p_adapter": cls.__name__,
                 **{f"p_{k}": v for k, v in params.items()}}
        lines.append((label, where))
    return lines


def make_test_configs(embedding_dim: int, device: str, train_n: list) -> list:
    """Expand the frozen set across the enrollment-budget grid.

    Mirrors the shape returned by `sweep()`: `(name, class, kwargs)` triples.
    The AE additionally needs `input_dim`/`device`; every family gets `train_n`.
    """
    configs = []
    for label, cls, params in FROZEN_BEST:
        extra = {}
        if cls is SmallAEAdapter:
            extra = {"input_dim": embedding_dim, "device": device}
        for n in train_n:
            kwargs = {**params, **extra, "train_n": n}
            tag = " ".join(f"{k}={v}" for k, v in params.items())
            name = f"{cls.__name__} {tag} train_n={n}".strip()
            configs.append((name, cls, kwargs))
    return configs
