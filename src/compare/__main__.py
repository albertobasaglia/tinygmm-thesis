"""
Comparison framework for one-class adapters (autoencoder, GMM, ...).

Usage:
    python -m src.compare

Edit CHECKPOINTS and make_configs() to control what gets compared.
Results are saved as a Parquet file in results/.
"""

import cProfile
import io
import logging
import pstats
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


def _upsert(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new_df into existing_df; new rows overwrite existing rows
    sharing the same p_-prefixed parameter columns.
    """
    if existing_df.empty:
        return new_df
    if new_df.empty:
        return existing_df

    p_cols = sorted(
        set(c for c in existing_df.columns if c.startswith("p_"))
        | set(c for c in new_df.columns if c.startswith("p_"))
    )
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    # keep last occurrence => new rows overwrite existing ones
    filled = combined[p_cols].fillna("__NA__")
    mask = ~filled.duplicated(keep="last")
    return combined[mask].reset_index(drop=True)


def main():
    PROFILE = False   # set True to run cProfile and dump results/profile.prof
    OVERWRITE = True  # True: run all configs and overwrite matching historical rows
                      # False: skip configs whose params already exist in the parquet

    logging.basicConfig(
        level=logging.DEBUG if PROFILE else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    N_TRIALS = 10
    MAX_TARGET_WORDS = None  # limit to first N target words (None = all)
    TEST_WORDS = {"visual", "five", "seven", "no", "off"}  # reserved for final evaluation, excluded from sweep

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
    if not existing_df.empty:
        log.info("Loaded %d existing rows from %s", len(existing_df), out_path)

    # Pre-compute skip lookup only when NOT overwriting
    if not OVERWRITE and not existing_df.empty:
        skip_p_cols = sorted(c for c in existing_df.columns if c.startswith("p_"))
        skip_keys = set(
            existing_df[skip_p_cols].fillna("__NA__").itertuples(index=False, name=None)
        )
    else:
        skip_p_cols = []
        skip_keys = set()

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
                "k": list(range(1, 11, 1)),
            }),
            # TODO: add ep=500 (or 1000) to find the true convergence floor.
            # At ep=200 the loss still drops ~18% in the last 20% of training,
            # so the AE has not fully converged. If ep=500 EER stays above
            # GMM's ~0.140, the GMM argument holds unconditionally.
            *sweep(SmallAEAdapter, {
                "train_n": train_n,
                "latent_dim": [4],
                "epochs": [30],
                "threshold_mode": ["val", "train"],
                "dropout_p": [0.0, 0.2],
                "device": [DEVICE],
                "input_dim": [embedding_dim],
            })
        ]

    # --- Loop over providers ---
    rows = []
    total_t0 = time.perf_counter()

    if PROFILE:
        _pr = cProfile.Profile()
        _pr.enable()

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

                if skip_keys:
                    extra = set(p_row.keys()) - set(skip_p_cols)
                    if not extra:
                        key = tuple(p_row.get(c, "__NA__") for c in skip_p_cols)
                        if key in skip_keys:
                            log.debug("Skipping existing config '%s' trial %d", name, trial)
                            continue

                log.info("Running config '%s'", name)
                config_t0 = time.perf_counter()
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

                config_t1 = time.perf_counter()
                log.info("Config '%s' took %.1fs", name, config_t1 - config_t0)

            log.info("Trial %d/%d done in %.1fs", trial + 1, N_TRIALS, time.perf_counter() - trial_t0)

    total_elapsed = time.perf_counter() - total_t0
    log.info("All experiments completed in %.1fs", total_elapsed)

    if PROFILE:
        _pr.disable()
        s = io.StringIO()
        pstats.Stats(_pr, stream=s).sort_stats("cumulative").print_stats(40)
        print(s.getvalue())
        prof_path = Path(__file__).parent.parent.parent / "results" / "profile.prof"
        _pr.dump_stats(prof_path)
        log.info("Profile saved to %s", prof_path)

    new_df = pd.DataFrame(rows)
    if OVERWRITE:
        df = _upsert(existing_df, new_df)
        log.info("Upserted %d existing + %d new = %d total rows",
                 len(existing_df), len(new_df), len(df))
    elif not existing_df.empty and not new_df.empty:
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
