"""
Final evaluation on the 5 held-out TEST_WORDS.

Usage:
    python -m src.compare.test_eval

Runs a small hardcoded set of final configs against each of the 5 test words,
with the other 4 test words as the adversarial pool. Writes to
results/test.parquet, separate from the validation sweep.parquet.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from embeddings.speech import SpeechEmbeddingProvider

from .adapters import GMMAdapter, SmallAEAdapter, SkipConfig
from .metrics import evaluate


log = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    N_TRIALS = 10
    TEST_WORDS = ["visual", "five", "seven", "no", "off"]

    ROOT = Path(__file__).parent.parent.parent
    ckpt_path = ROOT / "best_16.ckpt"
    embedding_dim = 16

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test.parquet"

    CONFIGS = [
        ("GMM-final", GMMAdapter, {
            "train_n": 50,
            "n_components": 1,
            "covariance_type": "diag",
        }),
        ("AE-final", SmallAEAdapter, {
            "train_n": 50,
            "latent_dim": 4,
            "epochs": 200,
            "device": DEVICE,
            "input_dim": embedding_dim,
        }),
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

    rows = []
    total_t0 = time.perf_counter()

    for provider in providers:
        log.info(
            "Provider %s | target=%s  adversarial=%s",
            provider.name, provider.target_class, provider.other_classes,
        )

        t0 = time.perf_counter()
        train_emb, test_target, test_other = provider.get_embeddings(
            max(k["train_n"] for _, _, k in CONFIGS), TEST_N
        )
        log.info("Embeddings loaded in %.1fs", time.perf_counter() - t0)

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(seed=trial)
            shuffled_emb = rng.permutation(train_emb)

            for name, cls, kwargs in CONFIGS:
                p_row = {
                    "p_split": "test",
                    "p_trial": trial,
                    "p_embedding_dim": embedding_dim,
                    "p_target_class": provider.target_class,
                    "p_other_classes": "|".join(sorted(provider.other_classes)),
                    "p_adapter": cls.__name__,
                    **{f"p_{k}": v for k, v in kwargs.items()},
                }

                log.info("Running config '%s' trial %d on target=%s",
                         name, trial, provider.target_class)
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
