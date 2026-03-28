"""
Comparison framework for one-class adapters (autoencoder, GMM, ...).

Usage:
    python -m src.compare

Edit CHECKPOINTS and make_configs() to control what gets compared.
Results are saved as a Parquet file in results/.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from embeddings.base import EmbeddingProvider
from embeddings.speech import SpeechEmbeddingProvider

from .adapters import AutoencoderAdapter, LinearAEAdapter, GMMAdapter, KNNAdapter
from .metrics import evaluate
from .sweep import sweep

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    N_TRIALS = 1
    MAX_TARGET_WORDS = 5  # limit to first N target words (None = all)
    ROOT = Path(__file__).parent.parent.parent   # repo root

    # =================================================================
    # EMBEDDING PROVIDERS
    #
    # Each provider wraps a dataset + feature extractor and returns
    # (train, test_target, test_other) embedding arrays.
    # The sweep runs independently per provider; results are tagged
    # with embedding_dim so they can be plotted together.
    # =================================================================
    ckpt_path = ROOT / "logs/speech_extractor/version_3/checkpoints/speech_extractor_emb16_seed42.ckpt"
    meta = torch.load(ckpt_path, weights_only=True)
    held_out = list(meta["hyper_parameters"].get("held_out_words") or [])
    if MAX_TARGET_WORDS is not None:
        held_out = held_out[:MAX_TARGET_WORDS]
    log.info("Target words: %s", held_out)

    providers: list[EmbeddingProvider] = [
        SpeechEmbeddingProvider(ckpt_path, 16, ROOT / "data",
                                target_class=w,
                                other_classes=[o for o in held_out if o != w],
                                device=DEVICE)
        for w in held_out
    ]

    # =================================================================
    # SWEEP CONFIG
    #
    # Adapter configs — two ways to add:
    #
    # 1. Single config (name, AdapterClass, kwargs):
    #        ("Autoencoder", AutoencoderAdapter, {"device": DEVICE})
    #
    # 2. Parameter sweep — expands all combinations automatically:
    #        *sweep(GMMAdapter, {
    #            "n_components": [1, 3, 5],
    #            "covariance_type": ["full", "diag"],
    #        })
    #    This produces 6 entries: n=1/full, n=1/diag, n=3/full, ...
    #
    # NOTE: do not set input_dim in AutoencoderAdapter here — it is
    # injected automatically from the provider's embedding_dim below.
    # =================================================================
    train_n = list(range(10, 51, 2))

    def make_configs(embedding_dim: int) -> list:
        return [
            *sweep(AutoencoderAdapter, {
                "train_n": train_n,
                "epochs": [25, 50, 75, 100],
                "device": [DEVICE],
                "input_dim": [embedding_dim],
            }),
            *sweep(LinearAEAdapter, {
                "train_n": train_n,
                "latent_dim": [4, 8],
                "epochs": [25, 50, 75, 100],
                "device": [DEVICE],
                "input_dim": [embedding_dim],
            }),
            *sweep(GMMAdapter, {
                "train_n": train_n,
                "n_components": [1, 2, 3],
                "covariance_type": ["diag"],
            }),
            *sweep(KNNAdapter, {
                "train_n": train_n,
                "k": [1, 2],
            }),
        ]

    # --- Loop over providers ---
    rows = []
    total_t0 = time.perf_counter()

    for provider in providers:
        embedding_dim = provider.embedding_dim
        log.info(
            "Provider %s (dim=%d) | target=%s  adversarial=%s",
            provider.name, embedding_dim,
            provider.target_class, provider.other_classes,
        )

        t0 = time.perf_counter()
        train_emb, test_target, test_other = provider.get_embeddings(max(train_n), TEST_N)
        log.info("Embeddings loaded in %.1fs", time.perf_counter() - t0)

        configs = make_configs(embedding_dim)
        log.info("%d configs × %d trials = %d runs", len(configs), N_TRIALS, len(configs) * N_TRIALS)

        for trial in range(N_TRIALS):
            trial_t0 = time.perf_counter()
            rng = np.random.default_rng(seed=trial)
            shuffled_emb = rng.permutation(train_emb)

            for name, cls, kwargs in configs:
                adapter = cls(**kwargs)
                adapter.fit(shuffled_emb)
                rows.append({
                    "p_trial": trial,
                    "p_embedding_dim": embedding_dim,
                    "p_target_class": provider.target_class,
                    "p_other_classes": "|".join(sorted(provider.other_classes)),
                    "p_adapter": cls.__name__,
                    **{f"p_{k}": v for k, v in kwargs.items()},
                    **evaluate(adapter, test_target, test_other),
                })

            log.info("Trial %d/%d done in %.1fs", trial + 1, N_TRIALS, time.perf_counter() - trial_t0)

    total_elapsed = time.perf_counter() - total_t0
    log.info("All experiments completed in %.1fs", total_elapsed)

    df = pd.DataFrame(rows)

    # --- Save results ---
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "sweep.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(df), out_path)

    return df


if __name__ == "__main__":
    main()
