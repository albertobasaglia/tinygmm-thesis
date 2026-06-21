"""
Comparison framework for one-class adapters (autoencoder, GMM, ...).

Usage:
    python -m src.compare

Edit CHECKPOINTS and make_configs() to control what gets compared.
Results are saved as a Parquet file in results/.
"""

import argparse
import cProfile
import importlib
import inspect
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
from embeddings.har import HAREmbeddingProvider

from .adapters import SkipConfig
from .metrics import evaluate

log = logging.getLogger(__name__)


def read_classes(path: Path) -> list[str]:
    """Read class identifiers (one per line) from a class-list file."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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

    parser = argparse.ArgumentParser(prog="python -m src.compare")
    parser.add_argument(
        "config",
        help="Config module under src.compare.configs (e.g. har_baseline)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Class-split group to personalize on. 'val' (default): validation "
             "classes + full config grid (the main sweep). 'test': held-out "
             "test classes + frozen best-of-each config list (the final test).",
    )
    args = parser.parse_args()
    cfg = importlib.import_module(f"src.compare.configs.{args.config}")

    PROVIDER = cfg.PROVIDER

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    N_TRIALS = 10
    MAX_TARGET_CLASSES = None  # limit to first N target classes (None = all)

    ROOT = Path(__file__).parent.parent.parent   # repo root
    CLASSES_DIR = Path(__file__).parent / "classes"

    ckpt_name = getattr(cfg, "CHECKPOINT", None)
    ckpt_path = ROOT / ckpt_name if ckpt_name else None

    # --- Output paths ---
    # Each run writes to a timestamped parquet (never overwritten).
    # `latest_path` is a symlink that always points to the most recent run
    # for this provider, used for incremental loading and for plotting.
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.split == "test":
        # Final-test run: a single stable file per dataset, matching the name the
        # exporter's section_final_test reads via --test-parquet (test_<dataset>).
        dataset = args.config.removesuffix("_baseline")
        out_path = results_dir / f"test_{dataset}.parquet"
        latest_path = out_path
    else:
        out_path = results_dir / f"sweep_{args.config}_{timestamp}.parquet"
        latest_path = results_dir / f"sweep_{args.config}_latest.parquet"

    # --- Resume / duplicate policy ---
    # RESUME_FROM:   parquet to continue from (rows are merged into the output).
    #                None = fresh run (no merge).
    # ON_DUPLICATE:  what to do with new configs whose p_* tuple already exists
    #                in RESUME_FROM:
    #                  "skip"    — don't recompute, keep the existing row
    #                  "replace" — recompute and overwrite the existing row
    # The final-test parquet (test_<dataset>) is regenerated from scratch each
    # run so a stale prior file cannot mask rows; the main sweep resumes as before.
    RESUME_FROM: Path | None = (
        None if args.split == "test"
        else (latest_path if latest_path.exists() else None)
    )
    ON_DUPLICATE: str = "skip"

    if ON_DUPLICATE not in ("skip", "replace"):
        raise ValueError(f"ON_DUPLICATE must be 'skip' or 'replace', got {ON_DUPLICATE!r}")

    # Extra parquets whose p_* rows should be treated as already-done (skipped),
    # but never merged into the output. Useful for cross-machine deduping.
    SKIP_FROM_PATHS: list[Path] = []

    if RESUME_FROM is not None:
        existing_df = pd.read_parquet(RESUME_FROM)
        log.info("Resuming from %s (%d existing rows, on_duplicate=%s)",
                 RESUME_FROM, len(existing_df), ON_DUPLICATE)
    else:
        existing_df = pd.DataFrame()
        log.info("Fresh run: no prior results found at %s", latest_path)

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
        from src.lib.data import download_pendigits

        PENDIGITS_CLASSES = [str(i) for i in range(10)]
        TEST_DIGITS = read_classes(CLASSES_DIR / "pendigits" / "test.txt")
        if args.split == "test":
            target_digits = list(TEST_DIGITS)
        else:
            target_digits = [d for d in PENDIGITS_CLASSES if d not in set(TEST_DIGITS)]
        if MAX_TARGET_CLASSES is not None:
            target_digits = target_digits[:MAX_TARGET_CLASSES]
        log.info("Split=%s | target digits: %s", args.split, target_digits)
        log.info("Test digits: %s", sorted(set(TEST_DIGITS)))

        pendigits_path = download_pendigits(ROOT / "data")

        providers = [
            TabularEmbeddingProvider(
                data_path=pendigits_path,
                label_column="class",
                target_class=d,
                other_classes=[o for o in target_digits if o != d],
            )
            for d in target_digits
        ]

    elif PROVIDER == "har":
        # --- HAR provider: WISDM-2019 watch (accel + gyro) + trained extractor ---
        TEST_SUBJECTS: set[int] = {int(s) for s in read_classes(CLASSES_DIR / "har" / "test.txt")}
        if ckpt_path is None:
            raise ValueError(f"Config {args.config!r} (PROVIDER='har') must set CHECKPOINT")
        meta = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        held_out: list[int] = [int(s) for s in (meta["hyper_parameters"].get("held_out_subjects") or [])]
        embedding_dim = int(meta["hyper_parameters"]["embedding_dim"])
        if args.split == "test":
            targets = sorted(TEST_SUBJECTS)
        else:
            targets = [s for s in held_out if s not in TEST_SUBJECTS]
        if MAX_TARGET_CLASSES is not None:
            targets = targets[:MAX_TARGET_CLASSES]
        log.info("Split=%s | target subjects: %s", args.split, targets)
        log.info("Test subjects: %s", sorted(TEST_SUBJECTS))

        providers = [
            HAREmbeddingProvider(ckpt_path, embedding_dim, ROOT / "data",
                                 target_class=s,
                                 other_classes=[o for o in targets if o != s],
                                 device=DEVICE)
            for s in targets
        ]

    elif PROVIDER == "speech":
        # --- Speech provider: Google Speech Commands + trained extractor ---
        TEST_WORDS = set(read_classes(CLASSES_DIR / "speech" / "test.txt"))
        if ckpt_path is None:
            raise ValueError(f"Config {args.config!r} (PROVIDER='speech') must set CHECKPOINT")
        meta = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        held_out = list(meta["hyper_parameters"].get("held_out_words") or [])
        embedding_dim = int(meta["hyper_parameters"]["embedding_dim"])
        if args.split == "test":
            targets = sorted(TEST_WORDS)
        else:
            targets = [w for w in held_out if w not in TEST_WORDS]
        if MAX_TARGET_CLASSES is not None:
            targets = targets[:MAX_TARGET_CLASSES]
        log.info("Split=%s | target words: %s", args.split, targets)
        log.info("Test words: %s", sorted(TEST_WORDS))

        providers = [
            SpeechEmbeddingProvider(ckpt_path, embedding_dim, ROOT / "data",
                                    target_class=w,
                                    other_classes=[o for o in targets if o != w],
                                    device=DEVICE)
            for w in targets
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
    train_n = cfg.TRAIN_N

    # 'val' uses the full grid (main sweep); 'test' uses the frozen
    # best-of-each shortlist shared with export_plots.BEST_LINES.
    _cfg_fn = cfg.make_test_configs if args.split == "test" else cfg.make_configs

    def make_configs(embedding_dim: int) -> list:
        return _cfg_fn(embedding_dim, DEVICE)

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
                # Standardization depends only on (trial, train_n); cache the
                # scaled (enroll, test_target, test_other) per budget so
                # configs sharing a train_n reuse one scaler fit.
                scaled_cache: dict = {}

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
                            "p_target_class": str(provider.target_class),
                            "p_other_classes": "|".join(sorted(str(c) for c in provider.other_classes)),
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
                        adapter_kwargs = dict(kwargs)
                        if "seed" in inspect.signature(cls.__init__).parameters:
                            adapter_kwargs["seed"] = trial
                        budget = kwargs.get("train_n")
                        if budget not in scaled_cache:
                            enroll_raw = (
                                shuffled_emb[:budget] if budget is not None
                                else shuffled_emb
                            )
                            scaled_cache[budget] = provider.standardize(
                                enroll_raw, test_target, test_other
                            )
                        enroll, tt, to = scaled_cache[budget]

                        adapter = cls(**adapter_kwargs)
                        try:
                            adapter.fit(enroll)
                        except SkipConfig as e:
                            log.warning("Skipping config '%s': %s", name, e)
                            continue
                        rows.append({
                            **p_row,
                            **evaluate(adapter, tt, to),
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

    # The main sweep keeps a `_latest` symlink; the final-test run writes a single
    # stable file (latest_path == out_path) and needs no symlink.
    if latest_path != out_path:
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(out_path.name)
        log.info("Updated symlink %s -> %s", latest_path, out_path.name)

    return df


if __name__ == "__main__":
    main()
