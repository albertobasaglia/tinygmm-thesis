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

from .adapters import AutoencoderAdapter, SmallAEAdapter, GMMAdapter, KNNAdapter, SkipConfig
from .metrics import evaluate
from .sweep import sweep

log = logging.getLogger(__name__)


def _existing_keys(df: pd.DataFrame) -> tuple[set[tuple], list[str]]:
    """Build a set of p_-column tuples for O(1) duplicate lookup.

    Returns (key_set, p_cols) where p_cols is the sorted list of
    p_-prefixed column names used to build each tuple.
    """
    if df.empty:
        return set(), []
    p_cols = sorted(c for c in df.columns if c.startswith("p_"))
    keys = set(
        df[p_cols].fillna("__NA__").itertuples(index=False, name=None)
    )
    return keys, p_cols


def _make_key(p_row: dict, p_cols: list[str]) -> tuple:
    """Build a lookup tuple matching the column order from _existing_keys."""
    return tuple(p_row.get(c, "__NA__") for c in p_cols)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    N_TRIALS = 10
    MAX_TARGET_WORDS = None  # limit to first N target words (None = all)
    TEST_WORDS = {"up", "visual", "wow", "yes", "zero"}  # reserved for final evaluation, excluded from sweep
    ROOT = Path(__file__).parent.parent.parent   # repo root

    # --- Load existing results (incremental mode) ---
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "sweep.parquet"
    # set True to ignore existing results and recompute everything

    FRESH = False

    if FRESH:
        existing_df = pd.DataFrame()
    else:
        existing_df = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    existing_keys, p_cols = _existing_keys(existing_df)
    if existing_keys:
        log.info("Loaded %d existing rows from %s", len(existing_df), out_path)

    # =================================================================
    # EMBEDDING PROVIDERS
    #
    # Each provider wraps a dataset + feature extractor and returns
    # (train, test_target, test_other) embedding arrays.
    # The sweep runs independently per provider; results are tagged
    # with embedding_dim so they can be plotted together.
    # =================================================================
    # ckpt_path = ROOT / "logs/speech_extractor/version_3/checkpoints/speech_extractor_emb16_seed42.ckpt"
    # ckpt_path = ROOT / "/Users/alberto/Gits/tinygmm/logs/speech_extractor/version_8/checkpoints/speech_extractor_emb8_seed42.ckpt"
    ckpt_path = ROOT / "best_16.ckpt"
    meta = torch.load(ckpt_path, weights_only=True)
    held_out = list(meta["hyper_parameters"].get("held_out_words") or [])
    if MAX_TARGET_WORDS is not None:
        held_out = held_out[:MAX_TARGET_WORDS]
    held_out = [w for w in held_out if w not in TEST_WORDS]
    log.info("Validation words: %s", held_out)
    log.info("Test words (excluded): %s", sorted(TEST_WORDS))

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
    # train_n = [5, 10, 20, 50, 100, 200, 500]
    train_n = list(range(5, 100, 10))

    def make_configs(embedding_dim: int) -> list:
        return [
            # *sweep(AutoencoderAdapter, {
            #     "train_n": train_n,
            #     "epochs": [25, 50, 75, 100],
            #     "device": [DEVICE],
            #     "input_dim": [embedding_dim],
            # }),
            # *sweep(SmallAEAdapter, {
            #     "train_n": train_n,
            #     "latent_dim": [4, 8],
            #     "epochs": [25, 50, 75, 100, 200, 500, 1000],
            #     "device": [DEVICE],
            #     "input_dim": [embedding_dim],
            # }),
            # *sweep(AutoencoderAdapter, {
            #     "train_n": train_n,
            #     "epochs": [10000],
            #     "device": [DEVICE],
            #     "input_dim": [embedding_dim],
            # }),
            # *sweep(SmallAEAdapter, {
            #     "train_n": train_n,
            #     "latent_dim": [8],
            #     "epochs": [100],
            #     "device": [DEVICE],
            #     "input_dim": [embedding_dim],
            # })
            *sweep(GMMAdapter, {
                "train_n": train_n,
                "n_components": [1, 2, 3],
                "covariance_type": ["diag", "full", "spherical"],
            }),
            *sweep(KNNAdapter, {
                "train_n": train_n,
                "k": [1, 2, 3, 4, 5],
            }),
            *sweep(SmallAEAdapter, {
                "train_n": train_n,
                "latent_dim": [4],
                "epochs": [10, 50, 100],
                "device": [DEVICE],
                "input_dim": [embedding_dim],
            })
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
                p_row = {
                    "p_trial": trial,
                    "p_embedding_dim": embedding_dim,
                    "p_target_class": provider.target_class,
                    "p_other_classes": "|".join(sorted(provider.other_classes)),
                    "p_adapter": cls.__name__,
                    **{f"p_{k}": v for k, v in kwargs.items()},
                }

                if p_cols and _make_key(p_row, p_cols) in existing_keys:
                    log.debug("Skipping existing config '%s' trial %d", name, trial)
                    continue

                log.info("Running config '%s'", name)
                config_t0 = time.perf_counter()
                adapter = cls(**kwargs)
                try:
                    adapter.fit(shuffled_emb)
                except SkipConfig as e:
                    log.info("Skipping config '%s': %s", name, e)
                    continue
                rows.append({
                    **p_row,
                    **evaluate(adapter, test_target, test_other),
                })

                config_t1 = time.perf_counter()
                log.info("Config '%s' took %.1fs", name, config_t1 - config_t0)

            log.info("Trial %d/%d done in %.1fs", trial + 1, N_TRIALS, time.perf_counter() - trial_t0)

    total_elapsed = time.perf_counter() - total_t0
    log.info("All experiments completed in %.1fs", total_elapsed)

    new_df = pd.DataFrame(rows)
    if not existing_df.empty and not new_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
        log.info("Merged %d existing + %d new = %d total rows",
                 len(existing_df), len(new_df), len(df))
    elif not existing_df.empty:
        df = existing_df
        log.info("No new rows computed; keeping %d existing rows", len(df))
    else:
        df = new_df

    # --- Save results ---
    df.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(df), out_path)

    return df


if __name__ == "__main__":
    main()
