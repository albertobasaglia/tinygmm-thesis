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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from embeddings.base import EmbeddingProvider
from embeddings.tabular import TabularEmbeddingProvider
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

    logging.basicConfig(
        level=logging.DEBUG if PROFILE else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    PROVIDER = "pendigits"  # one of: "pendigits", "speech"

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # used by commented AE configs in make_configs
    TEST_N = 500
    N_TRIALS = 10
    MAX_TARGET_CLASSES = None  # limit to first N target classes (None = all)

    ROOT = Path(__file__).parent.parent.parent   # repo root

    # --- Output paths ---
    # Each run writes to a timestamped parquet (never overwritten).
    # `latest_path` is a symlink that always points to the most recent run
    # for this provider, used for incremental loading and for plotting.
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"sweep_{PROVIDER}_{timestamp}.parquet"
    latest_path = results_dir / f"sweep_{PROVIDER}_latest.parquet"

    # --- Resume / duplicate policy ---
    # RESUME_FROM:   parquet to continue from (rows are merged into the output).
    #                None = fresh run (no merge).
    # ON_DUPLICATE:  what to do with new configs whose p_* tuple already exists
    #                in RESUME_FROM:
    #                  "skip"    — don't recompute, keep the existing row
    #                  "replace" — recompute and overwrite the existing row
    RESUME_FROM: Path | None = latest_path if latest_path.exists() else None
    ON_DUPLICATE: str = "skip"

    if ON_DUPLICATE not in ("skip", "replace"):
        raise ValueError(f"ON_DUPLICATE must be 'skip' or 'replace', got {ON_DUPLICATE!r}")

    # Extra parquets whose p_* rows should be treated as already-done (skipped),
    # but never merged into the output. Useful for cross-machine deduping.
    SKIP_FROM_PATHS: list[Path] = []

    if RESUME_FROM is not None:
        existing_df = pd.read_parquet(RESUME_FROM)
        log.info("Loaded %d existing rows from %s", len(existing_df), RESUME_FROM)
    else:
        existing_df = pd.DataFrame()

    # Build the skip lookup. Sources:
    #   - existing_df, only when ON_DUPLICATE == "skip" (otherwise we rerun it).
    #   - every parquet listed in SKIP_FROM_PATHS (always honoured).
    skip_dfs: list[pd.DataFrame] = []
    if ON_DUPLICATE == "skip" and not existing_df.empty:
        skip_dfs.append(existing_df)
    for p in SKIP_FROM_PATHS:
        extra_df = pd.read_parquet(p)
        log.info("Loaded %d skip rows from %s", len(extra_df), p)
        skip_dfs.append(extra_df)

    if skip_dfs:
        skip_df = pd.concat(skip_dfs, ignore_index=True, sort=False)
        skip_p_cols = sorted(c for c in skip_df.columns if c.startswith("p_"))
        skip_keys = set(
            skip_df[skip_p_cols].fillna("__NA__").itertuples(index=False, name=None)
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
    #
    # Select with PROVIDER above. Only one provider runs per output.
    # =================================================================
    providers: list[EmbeddingProvider]

    if PROVIDER == "pendigits":
        # --- Tabular provider: Pendigits (raw features, no feature extractor) ---
        PENDIGITS_CLASSES = [str(i) for i in range(10)]
        TEST_DIGITS = {"7", "9"}  # reserved for final eval, excluded from sweep
        target_digits = [d for d in PENDIGITS_CLASSES if d not in TEST_DIGITS]
        if MAX_TARGET_CLASSES is not None:
            target_digits = target_digits[:MAX_TARGET_CLASSES]
        log.info("Target digits: %s", target_digits)
        log.info("Test digits (excluded): %s", sorted(TEST_DIGITS))

        providers = [
            TabularEmbeddingProvider(
                data_path=ROOT / "data" / "pendigits.parquet",
                label_column="class",
                target_class=d,
                other_classes=[o for o in target_digits if o != d],
            )
            for d in target_digits
        ]

    elif PROVIDER == "speech":
        # --- Speech provider: Google Speech Commands + trained extractor ---
        TEST_WORDS = {"visual", "five", "seven", "no", "off"}  # reserved for final eval
        # ckpt_path = ROOT / "logs/speech_extractor/version_3/checkpoints/speech_extractor_emb16_seed42.ckpt"
        # ckpt_path = ROOT / "logs/speech_extractor/version_8/checkpoints/speech_extractor_emb8_seed42.ckpt"
        ckpt_path = ROOT / "best_16.ckpt"
        meta = torch.load(ckpt_path, weights_only=True)
        held_out = list(meta["hyper_parameters"].get("held_out_words") or [])
        if MAX_TARGET_CLASSES is not None:
            held_out = held_out[:MAX_TARGET_CLASSES]
        held_out = [w for w in held_out if w not in TEST_WORDS]
        log.info("Validation words: %s", held_out)
        log.info("Test words (excluded): %s", sorted(TEST_WORDS))

        providers = [
            SpeechEmbeddingProvider(ckpt_path, 16, ROOT / "data",
                                    target_class=w,
                                    other_classes=[o for o in held_out if o != w],
                                    device=DEVICE)
            for w in held_out
        ]

    else:
        raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")

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
            *sweep(SmallAEAdapter, {
                "train_n": train_n,
                "latent_dim": [4, 8],
                "epochs": [100],
                "device": [DEVICE],
                "input_dim": [embedding_dim],
            }),
            # TODO: add ep=500 (or 1000) to find the true convergence floor.
            # At ep=200 the loss still drops ~18% in the last 20% of training,
            # so the AE has not fully converged. If ep=500 EER stays above
            # GMM's ~0.140, the GMM argument holds unconditionally.
            # *sweep(SmallAEAdapter, {
            #     "train_n": train_n,
            #     "latent_dim": [4],
            #     "epochs": [30],
            #     "threshold_mode": ["val", "train"],
            #     "dropout_p": [0.0, 0.2],
            #     "device": [DEVICE],
            #     "input_dim": [embedding_dim],
            # })
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

        with logging_redirect_tqdm(), tqdm(
            total=N_TRIALS * len(configs),
            desc=provider.name,
            unit="run",
        ) as pbar:
            for trial in range(N_TRIALS):
                trial_t0 = time.perf_counter()
                rng = np.random.default_rng(seed=trial)
                shuffled_emb = rng.permutation(train_emb)

                for name, cls, kwargs in configs:
                    try:
                        pbar.set_postfix(
                            trial=f"{trial + 1}/{N_TRIALS}",
                            cfg=name[:30],
                            refresh=False,
                        )
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

                        log.debug("Running config '%s'", name)
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
                        log.debug("Config '%s' took %.1fs", name, config_t1 - config_t0)
                    finally:
                        pbar.update(1)

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

    # In "skip" mode, new_df has no overlap with existing_df (we skipped them),
    # so _upsert behaves like a concat. In "replace" mode, _upsert overwrites
    # matching rows. Both cases collapse to the same call.
    new_df = pd.DataFrame(rows)
    df = _upsert(existing_df, new_df)
    log.info("Merged %d existing + %d new = %d total rows",
             len(existing_df), len(new_df), len(df))

    # --- Save results ---
    # Write the timestamped run (immutable), then refresh the `_latest` symlink.
    df.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(df), out_path)

    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(out_path.name)
    log.info("Updated symlink %s -> %s", latest_path, out_path.name)

    return df


if __name__ == "__main__":
    main()
