"""
Export thesis plots to PDF.

Usage:
    # Quick local export
    python -m src.compare.export_plots results/sweep_speech_latest.parquet

    # Thesis export
    python -m src.compare.export_plots results/sweep_speech_latest.parquet \\
        --out tinygmm-tex/figures \\
        --test-parquet results/test_speech.parquet

Default --out is figures/<derived-name>, where <derived-name> is the
parquet stem with leading 'sweep_' and trailing '_latest' stripped.
The LaTeX summary table (section E) is written to <out>/../tables/.

Plot structure (matches the results chapter):
  A. Hyperparameter selection  (covariance, K, k, AE latent dim)
  B. Main adapter comparison   (best-of-each)
  C. Statistical confidence    (95% CI bar charts)
  D. Computational cost        (inference FLOPs + Pareto)
  E. Final test on held-out words (test_summary table + 2 figures) -- requires --test-parquet
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from .adapters import (
    CosineAdapter,
    GMMAdapter,
    KNNAdapter,
    PrototypeAdapter,
    SmallAEAdapter,
)
from .configs.frozen import best_lines
from .plots import _filter, plot_lines, plot_gmm_grid, plot_ci_bars

ROOT = Path(__file__).parent.parent.parent

FULL_W = 5.12
H = 3.0

plt.rcParams.update({
    "figure.figsize": (FULL_W, H),
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


# Single source of truth: the frozen best-of-each set lives in configs/frozen.py
# and is shared with the final-test sweep (make_test_configs) so they cannot drift.
BEST_LINES = best_lines()

GMM_COV_LINES = [
    ("K=1 diag", {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "diag"}),
    ("K=2 diag", {"p_adapter": "GMMAdapter", "p_n_components": 2, "p_covariance_type": "diag"}),
    ("K=3 diag", {"p_adapter": "GMMAdapter", "p_n_components": 3, "p_covariance_type": "diag"}),
    ("K=1 full", {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "full"}),
    ("K=2 full", {"p_adapter": "GMMAdapter", "p_n_components": 2, "p_covariance_type": "full"}),
    ("K=3 full", {"p_adapter": "GMMAdapter", "p_n_components": 3, "p_covariance_type": "full"}),
    ("K=1 sph",  {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "spherical"}),
    ("K=2 sph",  {"p_adapter": "GMMAdapter", "p_n_components": 2, "p_covariance_type": "spherical"}),
    ("K=3 sph",  {"p_adapter": "GMMAdapter", "p_n_components": 3, "p_covariance_type": "spherical"}),
]

AE_LINES = [
    ("L=4", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 100}),
    ("L=8", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 8, "p_epochs": 100}),
]

PARETO_LINES = [
    ("SmallAE",   {"p_adapter": "SmallAEAdapter"}),
    ("GMM",       {"p_adapter": "GMMAdapter"}),
    ("kNN",       {"p_adapter": "KNNAdapter"}),
    ("Cosine",    {"p_adapter": "CosineAdapter"}),
    ("Prototype", {"p_adapter": "PrototypeAdapter"}),
]

FIXED_TRAIN_N = int(os.getenv("TRAIN_N", "50"))

DATASET_PRETTY = {"speech": "Speech", "har": "HAR", "pendigits": "Pendigits"}


def _pretty_dataset(dataset: str) -> str:
    """Display name for a dataset key, e.g. 'har' -> 'HAR'."""
    return DATASET_PRETTY.get(dataset, dataset.replace("_", " ").title())


def _pareto_mask(x: np.ndarray, y: np.ndarray, lower_y_better: bool = True) -> np.ndarray:
    """Boolean mask for Pareto-optimal points.

    x is always minimized (FLOPs: lower is better). The y direction is set by
    lower_y_better: True for EER (lower better), False for ACC@FAR=5% (higher
    better). When higher y is preferred the comparison on y is flipped.
    """
    y_dir = y if lower_y_better else -y
    is_pareto = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominated = ((x <= x[i]) & (y_dir <= y_dir[i]) & ((x < x[i]) | (y_dir < y_dir[i])))
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False
    return is_pareto


def _save(out_dir: Path, name: str):
    plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  saved {name}.pdf")


def _dataset_name(parquet: Path) -> str:
    """Dataset key from the parquet stem (e.g. 'speech', 'har', 'pendigits').

    Mirrors _derive_out_dir: strip the leading 'sweep_'/trailing '_latest', then
    drop a trailing '_baseline' and any trailing timestamp tokens so all three
    datasets get a stable, collision-free namespace.
    """
    name = parquet.stem.removeprefix("sweep_").removesuffix("_latest")
    name = name.removesuffix("_baseline")
    parts = name.split("_")
    # Drop trailing timestamp-like tokens (all digits), e.g. 20260529_115500.
    while len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    name = "_".join(parts)
    return name.removesuffix("_baseline")


def _refresh_structural_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite m_inference_flops, m_parameters and m_training_flops with
    values recomputed from the current cost models in adapters.py.

    The sweep bakes these structural counts into the parquet at sweep time,
    so a later cost-model fix would silently leave the Pareto plots on the
    old formulas. Recomputing at export time keeps every exported artifact
    on the same accounting as export_resource and export_bench, which
    already compute live. As there, a synthetic fit only populates the
    shapes the cost models read; the counts do not depend on the data.

    The enrollment (fit) count m_training_flops needs one extra convention to
    stay consistent with export_resource's enrollment bar chart and the
    resource table: those report the GMM's *best-case* fit as a single
    closed-form pass (one weighted mean and covariance), so they set
    n_iter_=1 before reading training_flops(). The raw m_training_flops baked
    into the parquet instead reflects sklearn's actual EM (~2 iterations) and
    would disagree with fig:enroll_flops_bar. We mirror the n_iter_=1
    convention here so the enrollment Pareto agrees with both. The other
    adapters' fit costs scale only with the enrollment size (and the AE's
    epoch count), which the cost models read from the reconstructed adapter,
    so no fit is required for them; the GMM is fit only to populate the shapes
    its cost model reads.
    """
    rng = np.random.default_rng(0)
    cache: dict[tuple, tuple[int, int, int] | None] = {}

    def val(row, col):
        v = row.get(col)
        return None if v is None or pd.isna(v) else v

    def costs(row):
        # p_epochs joins the key because the AE's enrollment FLOPs scale with
        # the number of training epochs (inference FLOPs and parameters do not).
        key = (row["p_adapter"], val(row, "p_embedding_dim"),
               val(row, "p_n_components"), val(row, "p_covariance_type"),
               val(row, "p_k"), val(row, "p_train_n"), val(row, "p_latent_dim"),
               val(row, "p_epochs"))
        if key in cache:
            return cache[key]
        name, D, K, cov, k, n, L, epochs = key
        D = int(D)
        a = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ill-conditioned cov / convergence
            if name == "GMMAdapter":
                # Fit at the row's own enrollment size so the EM cost model sees
                # the right N; inference FLOPs and parameters do not depend on N.
                a = GMMAdapter(n_components=int(K), covariance_type=cov,
                               train_n=int(n), seed=0)
                a.fit(rng.standard_normal((int(n), D)).astype(np.float32))
                # Charge the best-case single closed-form pass, mirroring
                # export_resource so the enrollment Pareto agrees with
                # fig:enroll_flops_bar and tab:resource.
                a._gmm.n_iter_ = 1
            elif name == "KNNAdapter":
                a = KNNAdapter(k=int(k), train_n=int(n))
                a.fit(rng.standard_normal((int(n), D)).astype(np.float32))
            elif name == "SmallAEAdapter":
                # The AE cost model reads only constructor fields (train_n and
                # epochs included); _fitted_train_n falls back to train_n when
                # unfit, so no fit is needed for either FLOP count.
                a = SmallAEAdapter(input_dim=D, latent_dim=int(L),
                                   epochs=int(epochs), train_n=int(n))
            elif name == "CosineAdapter":
                # Fit at the row's enrollment size so the (mean over N) fit cost
                # is charged at the right N.
                a = CosineAdapter(train_n=int(n))
                a.fit(rng.standard_normal((int(n), D)).astype(np.float32))
            elif name == "PrototypeAdapter":
                a = PrototypeAdapter(train_n=int(n))
                a.fit(rng.standard_normal((int(n), D)).astype(np.float32))
        cache[key] = ((a.inference_flops(), a.parameters(), a.training_flops())
                      if a is not None else None)
        return cache[key]

    fresh = df.apply(costs, axis=1)
    known = fresh.notna()
    unknown = sorted(df.loc[~known, "p_adapter"].unique())
    if unknown:
        warnings.warn(f"no cost model for {unknown}; keeping baked values for those rows")
    df = df.copy()
    new_flops = fresh[known].map(lambda c: c[0])
    new_params = fresh[known].map(lambda c: c[1])
    new_train_flops = fresh[known].map(lambda c: c[2])
    stale_flops = int((df.loc[known, "m_inference_flops"] != new_flops).sum())
    stale_params = int((df.loc[known, "m_parameters"] != new_params).sum())
    stale_train = int((df.loc[known, "m_training_flops"] != new_train_flops).sum())
    df.loc[known, "m_inference_flops"] = new_flops
    df.loc[known, "m_parameters"] = new_params
    df.loc[known, "m_training_flops"] = new_train_flops
    if stale_flops or stale_params or stale_train:
        warnings.warn(
            "parquet was baked with outdated cost models: refreshed "
            f"{stale_flops} m_inference_flops, {stale_params} m_parameters and "
            f"{stale_train} m_training_flops values; the exported figures use "
            "the current formulas (the GMM enrollment count is the best-case "
            "single-pass fit, matching fig:enroll_flops_bar)"
        )
    else:
        print("Structural costs verified against current cost models (parquet already current)")
    return df


def _ci95(vals: pd.Series) -> tuple[float, float]:
    """Mean and 95% CI half-width across the rows (trial x target_class groups)."""
    vals = vals.dropna()
    n = len(vals)
    mean = vals.mean()
    if n < 2:
        return mean, float("nan")
    sem = vals.std(ddof=1) / np.sqrt(n)
    ci = stats.t.ppf(0.975, df=n - 1) * sem
    return mean, ci


def _cell(vals: pd.Series, fmt: str = ".3f", bold: bool = False) -> str:
    """A 'mean $\\pm$ ci' LaTeX cell at 95% CI; bold uses \\boldmath (not
    \\textbf, which does not bold math-mode content)."""
    mean, ci = _ci95(vals)
    s = f"${mean:{fmt}} \\pm {ci:{fmt}}$"
    return f"{{\\boldmath {s}}}" if bold else s


# Direction of "best" per metric column (higher better vs lower better).
METRIC_HIGHER_BETTER = {"m_acc_at_far5": True, "m_eer": False, "m_auprc": True}


def _best_rows(slices: list[pd.Series], metrics: list[str]) -> dict[str, int]:
    """Row index of the best value per metric: max mean where higher is better,
    else min. Used to bold the winning cell in each column."""
    best = {}
    for m in metrics:
        means = [s[m].mean() for s in slices]
        if not means:
            continue
        pick = max if METRIC_HIGHER_BETTER.get(m, True) else min
        best[m] = means.index(pick(means))
    return best


def section_hyperparam(df: pd.DataFrame, out_dir: Path):
    print("A. Hyperparameter selection")

    # EER versions (supporting / appendix material).
    plot_lines(df, x="p_train_n", y="m_eer", lines=GMM_COV_LINES,
               out_path=out_dir / "gmm_cov_eer.pdf",
               title="GMM: EER by covariance type")
    print("  saved gmm_cov_eer.pdf")

    plot_gmm_grid(df, train_n=FIXED_TRAIN_N, y="m_eer",
                  out_path=out_dir / "gmm_components_eer.pdf")
    print("  saved gmm_components_eer.pdf")

    plot_ci_bars(df, lines=GMM_COV_LINES, train_n=FIXED_TRAIN_N, y="m_eer",
                 out_path=out_dir / "gmm_cov_ci_eer.pdf",
                 title=f"GMM variants: EER with 95% CI (train_n={FIXED_TRAIN_N})",
                 xlabel="EER (lower is better)")
    print("  saved gmm_cov_ci_eer.pdf")

    # ACC@FAR=5% versions (headline metric).
    plot_lines(df, x="p_train_n", y="m_acc_at_far5", lines=GMM_COV_LINES,
               out_path=out_dir / "gmm_cov_acc_at_far5.pdf",
               title="GMM: ACC @ FAR=5% by covariance type",
               ylabel="ACC @ FAR=5%")
    print("  saved gmm_cov_acc_at_far5.pdf")

    plot_gmm_grid(df, train_n=FIXED_TRAIN_N, y="m_acc_at_far5",
                  out_path=out_dir / "gmm_components_acc_at_far5.pdf")
    print("  saved gmm_components_acc_at_far5.pdf")

    plot_ci_bars(df, lines=GMM_COV_LINES, train_n=FIXED_TRAIN_N, y="m_acc_at_far5",
                 out_path=out_dir / "gmm_cov_ci_acc_at_far5.pdf",
                 title=f"GMM variants: ACC @ FAR=5% with 95% CI (train_n={FIXED_TRAIN_N})",
                 xlabel="ACC @ FAR=5% (higher is better)")
    print("  saved gmm_cov_ci_acc_at_far5.pdf")

    knn_sub = _filter(df, {"p_adapter": "KNNAdapter"})
    knn_sub = knn_sub[knn_sub["p_train_n"] == FIXED_TRAIN_N] if "p_k" in knn_sub.columns else knn_sub.iloc[0:0]
    if knn_sub.empty:
        print("  skipped knn_k_selection (no KNN rows)")
    else:
        fig, ax = plt.subplots()
        agg = knn_sub.groupby("p_k")["m_eer"].agg(["mean", "std"]).reset_index().sort_values("p_k")
        ax.bar(agg["p_k"], agg["mean"], yerr=agg["std"], capsize=3)
        ax.set_xlabel("k")
        ax.set_ylabel("EER")
        ax.set_title(f"kNN: EER vs k (train_n={FIXED_TRAIN_N})")
        ax.set_xticks(agg["p_k"].astype(int))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(out_dir, "knn_k_selection")

    # AE latent-dim selection (epochs fixed at 100), headline ACC@FAR=5%.
    if _filter(df, {"p_adapter": "SmallAEAdapter"}).empty:
        print("  skipped ae_acc_at_far5 (no AE rows)")
    else:
        plot_lines(df, x="p_train_n", y="m_acc_at_far5", lines=AE_LINES,
                   out_path=out_dir / "ae_acc_at_far5.pdf",
                   title="AE: ACC @ FAR=5% by latent dim",
                   ylabel="ACC @ FAR=5%")
        print("  saved ae_acc_at_far5.pdf")


def section_compare(df: pd.DataFrame, out_dir: Path):
    print("B. Main comparison")

    plot_lines(df, x="p_train_n", y="m_eer", lines=BEST_LINES,
               out_path=out_dir / "compare_eer.pdf")
    print("  saved compare_eer.pdf")

    plot_lines(df, x="p_train_n", y="m_acc_at_far5", lines=BEST_LINES,
               out_path=out_dir / "compare_acc_at_far.pdf",
               title="Accuracy at FAR=5% vs enrollment budget",
               ylabel="ACC @ FAR=5%")
    print("  saved compare_acc_at_far.pdf")


def section_confidence(df: pd.DataFrame, out_dir: Path):
    print("C. Statistical confidence")

    plot_ci_bars(df, lines=BEST_LINES, train_n=FIXED_TRAIN_N, y="m_eer",
                 out_path=out_dir / "ci_eer.pdf",
                 title=f"EER with 95% CI (train_n={FIXED_TRAIN_N})",
                 xlabel="EER (lower is better)")
    print("  saved ci_eer.pdf")

    plot_ci_bars(df, lines=BEST_LINES, train_n=FIXED_TRAIN_N, y="m_acc_at_far5",
                 out_path=out_dir / "ci_acc_at_far5.pdf",
                 title=f"ACC @ FAR=5% with 95% CI (train_n={FIXED_TRAIN_N})",
                 xlabel="ACC @ FAR=5% (higher is better)")
    print("  saved ci_acc_at_far5.pdf")


def section_cost(df: pd.DataFrame, out_dir: Path, emit_acc_pareto: bool = True):
    print("D. Computational cost (Pareto)")

    # The dataset-independent resource artifacts (inference FLOPs bar, FLOPs vs
    # train_n, and the resource table) are generated once, structurally, by
    # src.compare.export_resource -- they depend only on the adapter and the
    # embedding dimension, not on the sweep, so they live there rather than
    # being re-emitted (and duplicated) per dataset here. Only the Pareto plots
    # belong here, because their accuracy axis is dataset-specific.

    # EER Pareto (lower is better) -- supporting material. Always from the
    # validation sweep, which carries the full per-adapter point clouds.
    _pareto_figure(df, out_dir, name="pareto_inference",
                   y="m_eer", ylabel="EER (lower is better)",
                   title="Pareto Frontier: EER vs Inference FLOPs",
                   lower_y_better=True)

    # ACC@FAR=5% Pareto (higher is better) -- headline metric. When a held-out
    # test parquet is supplied, section_final_test emits this from the TEST
    # accuracy of the frozen families instead, so the headline frontier matches
    # the test-led results. Skip it here in that case to avoid overwriting it
    # with the validation version.
    if emit_acc_pareto:
        _pareto_figure(df, out_dir, name="pareto_acc_at_far5",
                       y="m_acc_at_far5", ylabel="ACC @ FAR=5% (higher is better)",
                       title="Pareto Frontier: ACC @ FAR=5% vs Inference FLOPs",
                       lower_y_better=False)

    # Enrollment-cost counterparts: the same frontiers against the one-time fit
    # cost (m_training_flops, refreshed to the best-case single-pass GMM in
    # _refresh_structural_costs) instead of per-sample inference FLOPs. The EER
    # version is always emitted as supporting material.
    _pareto_figure(df, out_dir, name="pareto_enrollment_eer",
                   y="m_eer", ylabel="EER (lower is better)",
                   title="Pareto Frontier: EER vs Enrollment FLOPs",
                   lower_y_better=True,
                   cost_col="m_training_flops",
                   xlabel="Enrollment FLOPs (one-time)")

    # The validation ACC@FAR=5% enrollment Pareto is emitted only in the same
    # branch as its inference counterpart, so when a held-out test parquet is
    # supplied section_final_test overrides it with the test-led version
    # (matching the existing inference-Pareto pattern).
    if emit_acc_pareto:
        _pareto_figure(df, out_dir, name="pareto_acc_at_far5_enroll",
                       y="m_acc_at_far5", ylabel="ACC @ FAR=5% (higher is better)",
                       title="Pareto Frontier: ACC @ FAR=5% vs Enrollment FLOPs",
                       lower_y_better=False,
                       cost_col="m_training_flops",
                       xlabel="Enrollment FLOPs (one-time)")


def _pareto_axes(ax, ylabel: str, title: str, xlabel: str = "Inference FLOPs"):
    """Apply the shared Pareto axis style (log-x FLOPs, labels, grid).

    xlabel defaults to the inference variant; the enrollment Paretos pass
    "Enrollment FLOPs (one-time)" instead so the two cost axes are named
    consistently with the resource bar charts.
    """
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")


def _pareto_figure(df: pd.DataFrame, out_dir: Path, name: str, y: str,
                   ylabel: str, title: str, lower_y_better: bool,
                   cost_col: str = "m_inference_flops",
                   xlabel: str = "Inference FLOPs"):
    """Scatter of (cost, y) per adapter with its Pareto frontier.

    cost_col selects the x-axis cost: m_inference_flops (per-sample inference,
    the default) or m_training_flops (the one-time enrollment cost). x (cost)
    is always minimized; y is minimized when lower_y_better, else maximized
    (so the Pareto-optimal direction flips for ACC@FAR=5%).

    The cost axis is log-scaled, which has no zero, so any non-positive cost
    is dropped (the kNN baseline does no fitting arithmetic, so its enrollment
    FLOPs are 0). Dropped points are logged rather than silently capped, in
    keeping with the repo's no-silent-caps style.
    """
    fig, ax = plt.subplots()
    n_dropped = 0
    for label, where in PARETO_LINES:
        subset = _filter(df, where)
        if subset.empty:
            continue
        agg = subset.groupby(cost_col)[y].mean().reset_index()
        xs, ys = agg[cost_col].values.astype(float), agg[y].values
        positive = xs > 0
        n_dropped += int((~positive).sum())
        xs, ys = xs[positive], ys[positive]
        if len(xs) == 0:
            continue
        color = ax._get_lines.get_next_color()
        ax.scatter(xs, ys, alpha=0.4, s=10, color=color)
        pareto = _pareto_mask(xs, ys, lower_y_better=lower_y_better)
        if pareto.any():
            px, py = xs[pareto], ys[pareto]
            order = np.argsort(px)
            ax.scatter(px, py, s=35, color=color, label=label, zorder=3)
            ax.plot(px[order], py[order], color=color, linewidth=1.5, alpha=0.7, zorder=2)
    if n_dropped:
        print(f"  note: dropped {n_dropped} point(s) with non-positive {cost_col} "
              f"from {name} (log x-axis); typically kNN's zero-cost enrollment")
    _pareto_axes(ax, ylabel, title, xlabel=xlabel)
    fig.tight_layout()
    _save(out_dir, name)


def _pareto_families_figure(labels, xs, ys, out_dir: Path, name: str,
                            ylabel: str, title: str, lower_y_better: bool,
                            xlabel: str = "Inference FLOPs"):
    """Pareto scatter for one point per frozen family (label, cost, accuracy).

    Used for the headline ACC@FAR=5% frontier built from the held-out TEST
    accuracy of the six selected configs (one (cost, accuracy) point each).
    The cost is either inference or enrollment FLOPs; xlabel names which.
    Pareto-optimal points are highlighted and connected; dominated points are
    drawn faint. x (cost) is always minimized; y maximized for ACC@FAR=5%.

    The cost axis is log-scaled, so families with non-positive cost (e.g. the
    kNN baseline's zero enrollment FLOPs) are dropped and logged rather than
    silently capped.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    labels = list(labels)
    positive = xs > 0
    n_dropped = int((~positive).sum())
    if n_dropped:
        dropped = [labels[i] for i in range(len(labels)) if not positive[i]]
        print(f"  note: dropped {n_dropped} family/families with non-positive "
              f"cost from {name} (log x-axis): {dropped}")
        labels = [labels[i] for i in range(len(labels)) if positive[i]]
        xs, ys = xs[positive], ys[positive]
    if len(xs) == 0:
        print(f"  skipped {name}: no positive-cost families to plot")
        return
    pareto = _pareto_mask(xs, ys, lower_y_better=lower_y_better)

    fig, ax = plt.subplots()
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for i, label in enumerate(labels):
        color = palette[i % len(palette)]
        on = bool(pareto[i])
        ax.scatter(xs[i], ys[i], s=45 if on else 22,
                   color=color, alpha=1.0 if on else 0.45,
                   zorder=3 if on else 2,
                   edgecolors="black" if on else "none", linewidths=0.6,
                   label=label)
    # Connect the frontier points in increasing FLOPs.
    if pareto.any():
        order = np.argsort(xs[pareto])
        ax.plot(xs[pareto][order], ys[pareto][order],
                color="0.4", linewidth=1.3, alpha=0.7, zorder=1)
    _pareto_axes(ax, ylabel, title, xlabel=xlabel)
    fig.tight_layout()
    _save(out_dir, name)


def section_tables(df: pd.DataFrame, tables_dir: Path, dataset: str):
    """Emit the per-dataset booktabs tables (compare, gmm_ablation).

    Filenames are namespaced by dataset so the three datasets do not overwrite
    each other. Score cells report mean $\\pm$ 95% CI across the
    (trial x target_class) groups at train_n=FIXED_TRAIN_N. The resource table is
    dataset-independent and generated separately by src.compare.export_resource.
    """
    print("F. LaTeX tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["p_train_n"] == FIXED_TRAIN_N]

    _table_compare(sub, tables_dir, dataset)
    _table_gmm_ablation(sub, tables_dir, dataset)


def _table_with_ci(caption: str, label: str, tabular: str, ci_pdf: str,
                   table_frac: float = 0.46, plot_frac: float = 0.52) -> str:
    """A table float: the tabular (left) beside its CI plot (right).

    `tabular` is the full \\begin{tabular}..\\end{tabular} block; `ci_pdf` is a
    doc-relative path to the CI image (same convention as the chapter figures).
    The tabular is scaled to its minipage so narrow and wide tables both fit.
    """
    return "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{minipage}}[c]{{{table_frac}\\textwidth}}",
        "    \\centering",
        "    \\resizebox{\\linewidth}{!}{%",
        tabular,
        "    }",
        "  \\end{minipage}\\hfill",
        f"  \\begin{{minipage}}[c]{{{plot_frac}\\textwidth}}",
        "    \\centering",
        f"    \\includegraphics[width=\\linewidth]{{{ci_pdf}}}",
        "  \\end{minipage}",
        "\\end{table}",
    ])


def _table_compare(sub: pd.DataFrame, tables_dir: Path, dataset: str):
    """Best-of-each adapter rows, ACC@FAR=5% (primary) + EER (secondary),
    shown beside the validation CI chart of the same train_n=50 numbers."""
    present = [(label, _filter(sub, where)) for label, where in BEST_LINES]
    present = [(label, s) for label, s in present if not s.empty]
    metrics = ["m_acc_at_far5", "m_eer"]
    best = _best_rows([s for _, s in present], metrics)
    rows = []
    for i, (label, s) in enumerate(present):
        cells = " & ".join(_cell(s[m], bold=best.get(m) == i) for m in metrics)
        rows.append(f"      {label} & {cells} \\\\")
    tabular = "\n".join([
        "    \\begin{tabular}{lrr}",
        "      \\toprule",
        "      Adaptive layer & ACC@FAR=5\\% & EER \\\\",
        "      \\midrule",
        "\n".join(rows),
        "      \\bottomrule",
        "    \\end{tabular}",
    ])
    caption = (f"Adaptive-layer comparison on {_pretty_dataset(dataset)} at"
               f" \\texttt{{train\\_n}}={FIXED_TRAIN_N} (mean $\\pm$ 95\\% CI across"
               " target classes $\\times$ trials). ACC@FAR=5\\% is the headline"
               " metric (higher is better); EER is shown for reference (lower is"
               " better). The best value in each column is in bold. The same"
               " ACC@FAR=5\\% values are drawn as 95\\% confidence intervals at right.")
    tex = _table_with_ci(caption, f"tab:compare_{dataset}", tabular,
                         f"figures/{dataset}/ci_acc_at_far5.pdf")
    path = tables_dir / f"compare_{dataset}.tex"
    path.write_text(tex + "\n")
    print(f"  saved {path}")


def _table_gmm_ablation(sub: pd.DataFrame, tables_dir: Path, dataset: str):
    """K x covariance grid, ACC@FAR=5% mean +/- 95% CI, shown beside the ranked
    K x covariance CI chart of the same train_n=50 numbers."""
    g = _filter(sub, {"p_adapter": "GMMAdapter"})
    cov_types = [c for c in ("spherical", "diag", "full")
                 if c in set(g["p_covariance_type"].unique())]
    components = sorted(int(k) for k in g["p_n_components"].unique())
    # Best configuration in the grid (highest mean ACC@FAR=5%), bolded below.
    best_kc, best_mean = None, -np.inf
    for k in components:
        for cov in cov_types:
            s = g[(g["p_n_components"] == k) & (g["p_covariance_type"] == cov)]
            if not s.empty and s["m_acc_at_far5"].mean() > best_mean:
                best_mean, best_kc = s["m_acc_at_far5"].mean(), (k, cov)
    rows = []
    for k in components:
        cells = []
        for cov in cov_types:
            s = g[(g["p_n_components"] == k) & (g["p_covariance_type"] == cov)]
            if s.empty:
                cells.append("--")
            else:
                cells.append(_cell(s["m_acc_at_far5"], bold=(k, cov) == best_kc))
        rows.append(f"      $K={k}$ & " + " & ".join(cells) + " \\\\")
    tabular = "\n".join([
        f"    \\begin{{tabular}}{{l{'r' * len(cov_types)}}}",
        "      \\toprule",
        "      & " + " & ".join(f"\\texttt{{{c}}}" for c in cov_types) + " \\\\",
        "      \\midrule",
        "\n".join(rows),
        "      \\bottomrule",
        "    \\end{tabular}",
    ])
    caption = (f"GMM ablation on {_pretty_dataset(dataset)}: ACC@FAR=5\\% at"
               f" \\texttt{{train\\_n}}={FIXED_TRAIN_N} (mean $\\pm$ 95\\% CI) across"
               " the number of components $K$ and covariance type; higher is"
               " better. The best configuration is in bold. The same values are"
               " drawn as ranked 95\\% confidence intervals at right.")
    tex = _table_with_ci(caption, f"tab:gmm_ablation_{dataset}", tabular,
                         f"figures/{dataset}/gmm_cov_ci_acc_at_far5.pdf",
                         table_frac=0.50, plot_frac=0.48)
    path = tables_dir / f"gmm_ablation_{dataset}.tex"
    path.write_text(tex + "\n")
    print(f"  saved {path}")


def section_final_test(test_parquet_path: Path | None, out_dir: Path,
                       tables_dir: Path, dataset: str):
    """Per-dataset final-test artifacts from the held-out test parquet.

    Emits a per-dataset summary table (`test_summary_<dataset>.tex`) and two
    figures (`test_<dataset>_acc_at_far5_ci`, `test_<dataset>_acc_at_far5_vs_train_n`)
    at the same enrollment budget (train_n=FIXED_TRAIN_N) used by the rest of the
    exporter. The headline metric is ACC@FAR=5% (higher is better); EER is
    reported in the table as a supporting column.

    Rows are the frozen best-of-each families (BEST_LINES), so the two GMM K=1
    variants (full vs diag) stay distinct rather than collapsing on p_adapter.
    """
    if test_parquet_path is None:
        print("E. Final test skipped (no --test-parquet)")
        return
    if not test_parquet_path.exists():
        print(f"E. Final test skipped ({test_parquet_path} not found)")
        return
    print(f"E. Final test ({dataset})")
    test_df = pd.read_parquet(test_parquet_path)
    # main() refreshes only the validation sweep df; the test parquet is read
    # here independently, so refresh its structural costs too. This is what
    # makes the test-led enrollment Pareto below charge the same best-case
    # single-pass GMM fit as fig:enroll_flops_bar (m_training_flops is baked
    # at sweep time with sklearn's ~2-iteration EM otherwise).
    test_df = _refresh_structural_costs(test_df)

    slice_at = test_df[test_df["p_train_n"] == FIXED_TRAIN_N]
    if slice_at.empty:
        print(f"  skipped: no rows at train_n={FIXED_TRAIN_N}")
        return

    # Each frozen family becomes one row/bar; sort by headline ACC@FAR=5% (desc).
    families = []  # (label, where, slice)
    for label, where in BEST_LINES:
        s = _filter(slice_at, where)
        if s.empty:
            continue
        families.append((label, where, s))
    families.sort(key=lambda t: t[2]["m_acc_at_far5"].mean(), reverse=True)
    if not families:
        print("  skipped: no frozen families present in test parquet")
        return

    tables_dir.mkdir(parents=True, exist_ok=True)
    n_targets = slice_at["p_target_class"].nunique()
    n_trials = slice_at["p_trial"].nunique()

    # --- Summary table (headline ACC@FAR=5% first, then EER), shown side by side
    #     with the CI chart of the same train_n=50 numbers (left: tabular, right:
    #     plot). The CI pdf is emitted just below; path is doc-relative like the
    #     chapter's other \includegraphics. ---
    metric_labels = {"m_acc_at_far5": "ACC@FAR=5\\%", "m_eer": "EER"}
    metrics = list(metric_labels)
    best = _best_rows([s for _, _, s in families], metrics)
    rows = []
    for i, (label, where, s) in enumerate(families):
        cells = " & ".join(_cell(s[m], bold=best.get(m) == i) for m in metrics)
        rows.append(f"      {label} & {cells} \\\\")
    ci_pdf = f"figures/{dataset}/test_{dataset}_acc_at_far5_ci.pdf"
    tabular = "\n".join([
        f"    \\begin{{tabular}}{{l{'r' * len(metrics)}}}",
        "      \\toprule",
        f"      {_pretty_dataset(dataset)} & "
        + " & ".join(metric_labels[m] for m in metrics) + " \\\\",
        "      \\midrule",
        "\n".join(rows),
        "      \\bottomrule",
        "    \\end{tabular}",
    ])
    caption = (f"Final-test metrics on the held-out {_pretty_dataset(dataset)} test classes"
               f" at \\texttt{{train\\_n}}={FIXED_TRAIN_N} (mean $\\pm$ 95\\% CI across"
               f" {n_targets} test classes $\\times$ {n_trials} trials). ACC@FAR=5\\% is"
               " the headline metric (higher is better); EER is shown for reference. The"
               " best value in each column is in bold. The same ACC@FAR=5\\% values are"
               " drawn as 95\\% confidence intervals at right.")
    tex = _table_with_ci(caption, f"tab:test_summary_{dataset}", tabular, ci_pdf)
    path = tables_dir / f"test_summary_{dataset}.tex"
    path.write_text(tex + "\n")
    print(f"  saved {path}")

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    # --- Headline chart: ACC@FAR=5% with 95% CI ---
    # Horizontal error-bar dot chart via plot_ci_bars, in the same best-first
    # order as the summary table it sits beside (families is sorted by mean ACC).
    ordered_lines = [(label, where) for label, where, _ in families]
    plot_ci_bars(test_df, lines=ordered_lines, train_n=FIXED_TRAIN_N,
                 y="m_acc_at_far5",
                 out_path=out_dir / f"test_{dataset}_acc_at_far5_ci.pdf",
                 title=f"{_pretty_dataset(dataset)}: ACC @ FAR=5% (train_n={FIXED_TRAIN_N}, 95% CI)",
                 xlabel="ACC @ FAR=5% (higher is better)")

    # --- Headline trend: ACC@FAR=5% vs enrollment budget ---
    fig, ax = plt.subplots()
    for (label, where, _), color in zip(families, palette):
        sub = _filter(test_df, where)
        agg_tn = (sub.groupby("p_train_n")["m_acc_at_far5"]
                  .agg(["mean", "std", "count"]).reset_index().sort_values("p_train_n"))
        ci = 1.96 * agg_tn["std"] / np.sqrt(agg_tn["count"])
        ax.plot(agg_tn["p_train_n"], agg_tn["mean"], marker="o",
                label=label, color=color)
        ax.fill_between(agg_tn["p_train_n"],
                        agg_tn["mean"] - ci, agg_tn["mean"] + ci,
                        alpha=0.15, color=color)
    ax.set_xlabel("Enrollment size (train_n)")
    ax.set_ylabel("ACC @ FAR=5%")
    ax.set_title(f"Final-test ACC @ FAR=5% vs enrollment budget "
                 f"({_pretty_dataset(dataset)})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(out_dir, f"test_{dataset}_acc_at_far5_vs_train_n")

    # --- Headline ACC@FAR=5% Pareto (TEST accuracy vs inference FLOPs) ---
    # One point per frozen family: cost is the (dataset-independent) inference
    # FLOPs, accuracy is the held-out test mean. This replaces the validation
    # Pareto so the headline frontier matches the test-led results chapter.
    pareto_labels, pareto_x, pareto_y = [], [], []
    for label, _, s in families:
        pareto_labels.append(label)
        pareto_x.append(float(s["m_inference_flops"].iloc[0]))
        pareto_y.append(float(s["m_acc_at_far5"].mean()))
    _pareto_families_figure(
        pareto_labels, pareto_x, pareto_y, out_dir,
        name="pareto_acc_at_far5",
        ylabel="ACC @ FAR=5% (higher is better)",
        title="Pareto Frontier: ACC @ FAR=5% vs Inference FLOPs (held-out test)",
        lower_y_better=False)

    # --- Headline ACC@FAR=5% enrollment Pareto (TEST accuracy vs fit cost) ---
    # Same frozen families and accuracy, but the cost axis is the one-time
    # enrollment (fit) FLOPs rather than per-sample inference FLOPs. The
    # m_training_flops column was refreshed above to the best-case single-pass
    # GMM fit, so this frontier agrees with fig:enroll_flops_bar and the
    # resource table. The kNN family has zero enrollment FLOPs and is dropped
    # from the log-scaled cost axis (logged by _pareto_families_figure).
    enroll_x = [float(s["m_training_flops"].iloc[0]) for _, _, s in families]
    _pareto_families_figure(
        pareto_labels, enroll_x, pareto_y, out_dir,
        name="pareto_acc_at_far5_enroll",
        ylabel="ACC @ FAR=5% (higher is better)",
        title="Pareto Frontier: ACC @ FAR=5% vs Enrollment FLOPs (held-out test)",
        lower_y_better=False,
        xlabel="Enrollment FLOPs (one-time)")

    # --- Paired GMM-vs-AE significance on the headline metric ---
    gmm_where = {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "diag"}
    ae_where = {"p_adapter": "SmallAEAdapter"}
    gmm_s = _filter(slice_at, gmm_where)
    ae_s = _filter(slice_at, ae_where)
    if not gmm_s.empty and not ae_s.empty:
        idx = ["p_trial", "p_target_class"]
        gmm_acc = gmm_s.set_index(idx)["m_acc_at_far5"]
        ae_acc = ae_s.set_index(idx)["m_acc_at_far5"]
        paired = pd.concat([gmm_acc.rename("gmm"), ae_acc.rename("ae")], axis=1).dropna()
        if len(paired) >= 2:
            diff = paired["gmm"] - paired["ae"]
            t, p = stats.ttest_rel(paired["gmm"], paired["ae"])
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else float("nan")
            print(f"  paired t-test (GMM - AE) ACC@FAR=5% at train_n={FIXED_TRAIN_N}: "
                  f"n={len(paired)}  mean={diff.mean():+.4f}  d={d:+.3f}  "
                  f"t={t:+.3f}  p={p:.4g}")


def print_headline_table(df: pd.DataFrame):
    print()
    print(f"Headline numbers @ train_n={FIXED_TRAIN_N} (mean over 10 trials x 10 target classes):")
    print()
    sub = df[df["p_train_n"] == FIXED_TRAIN_N]
    fmt = "  {:<14} {:>8} {:>8} {:>10} {:>8}"
    print(fmt.format("Adapter", "EER", "ACC@5%", "InfFLOPs", "Params"))
    print(fmt.format("-------", "---", "------", "--------", "------"))
    for label, where in BEST_LINES:
        s = _filter(sub, where)
        if s.empty:
            continue
        print(fmt.format(
            label,
            f"{s['m_eer'].mean():.3f}",
            f"{s['m_acc_at_far5'].mean():.3f}",
            f"{s['m_inference_flops'].mean():.0f}",
            f"{s['m_parameters'].mean():.0f}",
        ))


def _derive_out_dir(parquet: Path) -> Path:
    name = parquet.stem.removeprefix("sweep_").removesuffix("_latest")
    return ROOT / "figures" / name


def main():
    parser = argparse.ArgumentParser(prog="python -m src.compare.export_plots")
    parser.add_argument("parquet", type=Path,
                        help="Path to the sweep parquet (e.g. results/sweep_speech_latest.parquet)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Figures root. PDFs land in <out>/<dataset>; tables in "
                             "<out>/../tables. Defaults to figures/<derived-name>.")
    parser.add_argument("--test-parquet", type=Path, default=None,
                        help="Optional held-out test parquet. Enables section E (final test).")
    args = parser.parse_args()

    dataset = _dataset_name(args.parquet)
    if args.out is not None:
        # <out> is the figures root; PDFs go in a per-dataset subdir and the
        # LaTeX tables land in the shared sibling tables/ directory, so the
        # three datasets never overwrite each other.
        out_dir = args.out / dataset
        tables_dir = args.out.parent / "tables"
    else:
        out_dir = _derive_out_dir(args.parquet)
        tables_dir = out_dir.parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    df = _refresh_structural_costs(df)
    print(f"Loaded {len(df)} rows from {args.parquet}")
    print(f"Writing PDFs to {out_dir}")
    print(f"Writing tables to {tables_dir}\n")

    # When a held-out test parquet is supplied, the headline ACC@FAR=5% Pareto
    # is emitted from TEST accuracy by section_final_test, so section_cost must
    # not overwrite it with the validation version. The supporting EER Pareto
    # stays on the validation sweep regardless.
    have_test = args.test_parquet is not None and args.test_parquet.exists()

    section_hyperparam(df, out_dir)
    section_compare(df, out_dir)
    section_confidence(df, out_dir)
    section_cost(df, out_dir, emit_acc_pareto=not have_test)
    section_tables(df, tables_dir, dataset)
    section_final_test(args.test_parquet, out_dir, tables_dir, dataset)
    print_headline_table(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
