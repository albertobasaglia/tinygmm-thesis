"""
Final evaluation on the held-out test classes.

Usage:
    python -m src.compare.test_eval

Runs a small hardcoded set of final configs against each held-out test
class, with the other test classes as the adversarial pool. Writes to
results/test_<provider>.parquet, separate from the validation sweep.
Switch PROVIDER below to toggle between speech and pendigits.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from embeddings.speech import SpeechEmbeddingProvider
from embeddings.tabular import TabularEmbeddingProvider

from .adapters import (
    GMMAdapter, SmallAEAdapter, CosineAdapter, PrototypeAdapter, KNNAdapter,
    SkipConfig,
)
from .metrics import evaluate


log = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    PROVIDER = "pendigits"  # one of: "pendigits", "speech"

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    N_TRIALS = 10

    ROOT = Path(__file__).parent.parent.parent
    embedding_dim = 16

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"test_{PROVIDER}.parquet"

    TRAIN_N_VALUES = list(range(5, 100, 10))

    if PROVIDER == "speech":
        ckpt_path = ROOT / "best_16.ckpt"
        TEST_WORDS = ["visual", "five", "seven", "no", "off"]

        CONFIGS = [
            ("GMM-final", GMMAdapter, {
                "n_components": 1,
                "covariance_type": "diag",
            }),
            ("AE-final", SmallAEAdapter, {
                "latent_dim": 8,
                "epochs": 100,
                "device": DEVICE,
                "input_dim": embedding_dim,
            }),
            ("Cosine-final", CosineAdapter, {}),
            ("Prototype-final", PrototypeAdapter, {}),
        ]

        providers = [
            SpeechEmbeddingProvider(
                ckpt_path, embedding_dim, ROOT / "data",
                target_class=w,
                other_classes=[o for o in TEST_WORDS if o != w],
                device=DEVICE,
            )
            for w in TEST_WORDS
        ]

    elif PROVIDER == "pendigits":
        TEST_DIGITS = ["7", "9"]

        CONFIGS = [
            # Smallest GMM that ties best AUC on validation:
            ("GMM-K1-diag",  GMMAdapter, {"n_components": 1, "covariance_type": "diag"}),
            ("GMM-K2-sph",   GMMAdapter, {"n_components": 2, "covariance_type": "spherical"}),
            ("GMM-K1-full",  GMMAdapter, {"n_components": 1, "covariance_type": "full"}),
            ("AE-final",     SmallAEAdapter, {
                "latent_dim": 4, "epochs": 30,
                "threshold_mode": "train", "dropout_p": 0.0,
                "device": DEVICE, "input_dim": embedding_dim,
            }),
            ("Cosine-final",    CosineAdapter, {}),
            ("Prototype-final", PrototypeAdapter, {}),
            ("kNN-k5",          KNNAdapter, {"k": 5}),
        ]

        providers = [
            TabularEmbeddingProvider(
                data_path=ROOT / "data" / "pendigits.parquet",
                label_column="class",
                target_class=d,
                other_classes=[o for o in TEST_DIGITS if o != d],
            )
            for d in TEST_DIGITS
        ]

    else:
        raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")

    rows = []
    total_t0 = time.perf_counter()

    for provider in providers:
        log.info(
            "Provider %s | target=%s  adversarial=%s",
            provider.name, provider.target_class, provider.other_classes,
        )

        t0 = time.perf_counter()
        train_emb, test_target, test_other = provider.get_embeddings(
            max(TRAIN_N_VALUES), TEST_N
        )
        log.info("Embeddings loaded in %.1fs", time.perf_counter() - t0)

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(seed=trial)
            shuffled_emb = rng.permutation(train_emb)

            for train_n in TRAIN_N_VALUES:
                for name, cls, base_kwargs in CONFIGS:
                    kwargs = {**base_kwargs, "train_n": train_n}
                    p_row = {
                        "p_split": "test",
                        "p_trial": trial,
                        "p_embedding_dim": embedding_dim,
                        "p_target_class": provider.target_class,
                        "p_other_classes": "|".join(sorted(provider.other_classes)),
                        "p_adapter": cls.__name__,
                        **{f"p_{k}": v for k, v in kwargs.items()},
                    }

                    log.info("Running config '%s' trial %d train_n=%d target=%s",
                             name, trial, train_n, provider.target_class)
                    adapter = cls(**kwargs)
                    try:
                        adapter.fit(shuffled_emb)
                    except SkipConfig as e:
                        log.warning("Skipping config '%s': %s", name, e)
                        continue
                    rows.append({
                        **p_row,
                        **evaluate(adapter, test_target, test_other),
                    })

    log.info("All experiments completed in %.1fs", time.perf_counter() - total_t0)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(df), out_path)

    return df


if __name__ == "__main__":
    main()
