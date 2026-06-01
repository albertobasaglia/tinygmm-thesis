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
  A. Hyperparameter selection  (covariance, K, k, AE dropout)
  B. Main adapter comparison   (best-of-each)
  C. Statistical confidence    (95% CI bar charts)
  D. Computational cost        (inference FLOPs + Pareto)
  E. Final test on held-out words (test_summary table + 2 figures) -- requires --test-parquet
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

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


GMM_BEST       = {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "diag"}
KNN_BEST       = {"p_adapter": "KNNAdapter", "p_k": 5}
AE_BEST        = {"p_adapter": "SmallAEAdapter", "p_latent_dim": 8, "p_epochs": 100}
COSINE_BEST    = {"p_adapter": "CosineAdapter"}
PROTOTYPE_BEST = {"p_adapter": "PrototypeAdapter"}

BEST_LINES = [
    ("GMM K=1 diag", GMM_BEST),
    ("GMM K=1 full", {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "full"}),
    ("kNN k=5",      KNN_BEST),
    ("SmallAE",      AE_BEST),
    ("Cosine",       COSINE_BEST),
    ("Prototype",    PROTOTYPE_BEST),
]

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

PARETO_LINES = [
    ("SmallAE",   {"p_adapter": "SmallAEAdapter"}),
    ("GMM",       {"p_adapter": "GMMAdapter"}),
    ("kNN",       {"p_adapter": "KNNAdapter"}),
    ("Cosine",    {"p_adapter": "CosineAdapter"}),
    ("Prototype", {"p_adapter": "PrototypeAdapter"}),
]

FIXED_TRAIN_N = int(os.getenv("TRAIN_N", "50"))


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


def _cell(vals: pd.Series, fmt: str = ".3f") -> str:
    """A 'mean $\\pm$ ci' LaTeX cell at 95% CI."""
    mean, ci = _ci95(vals)
    return f"${mean:{fmt}} \\pm {ci:{fmt}}$"


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

    if "p_dropout_p" not in df.columns:
        print("  skipped ae_dropout_eer (no AE dropout rows)")
    else:
        dropout_lines = [
            ("AE no-dropout",    {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                                  "p_dropout_p": 0.0}),
            ("AE dropout_p=0.2", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                                  "p_dropout_p": 0.2}),
        ]
        plot_lines(df, x="p_train_n", y="m_eer", lines=dropout_lines,
                   out_path=out_dir / "ae_dropout_eer.pdf",
                   title="AE dropout ablation")
        print("  saved ae_dropout_eer.pdf")


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


def section_cost(df: pd.DataFrame, out_dir: Path):
    print("D. Computational cost (Pareto)")

    # The dataset-independent resource artifacts (inference FLOPs bar, FLOPs vs
    # train_n, and the resource table) are generated once, structurally, by
    # src.compare.export_resource -- they depend only on the adapter and the
    # embedding dimension, not on the sweep, so they live there rather than
    # being re-emitted (and duplicated) per dataset here. Only the Pareto plots
    # belong here, because their accuracy axis is dataset-specific.

    # EER Pareto (lower is better) -- supporting material.
    _pareto_figure(df, out_dir, name="pareto_inference",
                   y="m_eer", ylabel="EER (lower is better)",
                   title="Pareto Frontier: EER vs Inference FLOPs",
                   lower_y_better=True)

    # ACC@FAR=5% Pareto (higher is better) -- headline metric.
    _pareto_figure(df, out_dir, name="pareto_acc_at_far5",
                   y="m_acc_at_far5", ylabel="ACC @ FAR=5% (higher is better)",
                   title="Pareto Frontier: ACC @ FAR=5% vs Inference FLOPs",
                   lower_y_better=False)


def _pareto_figure(df: pd.DataFrame, out_dir: Path, name: str, y: str,
                   ylabel: str, title: str, lower_y_better: bool):
    """Scatter of (inference FLOPs, y) per adapter with its Pareto frontier.

    x (FLOPs) is always minimized; y is minimized when lower_y_better, else
    maximized (so the Pareto-optimal direction flips for ACC@FAR=5%).
    """
    fig, ax = plt.subplots()
    for label, where in PARETO_LINES:
        subset = _filter(df, where)
        if subset.empty:
            continue
        agg = subset.groupby("m_inference_flops")[y].mean().reset_index()
        xs, ys = agg["m_inference_flops"].values, agg[y].values
        color = ax._get_lines.get_next_color()
        ax.scatter(xs, ys, alpha=0.4, s=20, color=color)
        pareto = _pareto_mask(xs, ys, lower_y_better=lower_y_better)
        if pareto.any():
            px, py = xs[pareto], ys[pareto]
            order = np.argsort(px)
            ax.scatter(px, py, s=80, color=color, label=label, zorder=3)
            ax.plot(px[order], py[order], color=color, linewidth=1.5, alpha=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Inference FLOPs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
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


def _table_compare(sub: pd.DataFrame, tables_dir: Path, dataset: str):
    """Best-of-each adapter rows, ACC@FAR=5% (primary) + EER (secondary)."""
    rows = []
    for label, where in BEST_LINES:
        s = _filter(sub, where)
        if s.empty:
            continue
        rows.append(f"    {label} & {_cell(s['m_acc_at_far5'])} & {_cell(s['m_eer'])} \\\\")
    tex = "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Adaptive-layer comparison on {dataset} at"
        f" \\texttt{{train\\_n}}={FIXED_TRAIN_N} (mean $\\pm$ 95\\% CI across"
        " target classes $\\times$ trials). ACC@FAR=5\\% is the headline"
        " metric (higher is better); EER is shown for reference (lower is"
        " better).}",
        f"  \\label{{tab:compare_{dataset}}}",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        "    Adaptive layer & ACC@FAR=5\\% & EER \\\\",
        "    \\midrule",
        "\n".join(rows),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    path = tables_dir / f"compare_{dataset}.tex"
    path.write_text(tex + "\n")
    print(f"  saved {path}")


def _table_gmm_ablation(sub: pd.DataFrame, tables_dir: Path, dataset: str):
    """K x covariance grid, ACC@FAR=5% mean +/- 95% CI."""
    g = _filter(sub, {"p_adapter": "GMMAdapter"})
    cov_types = [c for c in ("spherical", "diag", "full")
                 if c in set(g["p_covariance_type"].unique())]
    components = sorted(int(k) for k in g["p_n_components"].unique())
    rows = []
    for k in components:
        cells = []
        for cov in cov_types:
            s = g[(g["p_n_components"] == k) & (g["p_covariance_type"] == cov)]
            cells.append(_cell(s["m_acc_at_far5"]) if not s.empty else "--")
        rows.append(f"    $K={k}$ & " + " & ".join(cells) + " \\\\")
    tex = "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{GMM ablation on {dataset}: ACC@FAR=5\\% at"
        f" \\texttt{{train\\_n}}={FIXED_TRAIN_N} (mean $\\pm$ 95\\% CI) across"
        " the number of components $K$ and covariance type. Higher is"
        " better.}",
        f"  \\label{{tab:gmm_ablation_{dataset}}}",
        f"  \\begin{{tabular}}{{l{'r' * len(cov_types)}}}",
        "    \\toprule",
        "    & " + " & ".join(f"\\texttt{{{c}}}" for c in cov_types) + " \\\\",
        "    \\midrule",
        "\n".join(rows),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    path = tables_dir / f"gmm_ablation_{dataset}.tex"
    path.write_text(tex + "\n")
    print(f"  saved {path}")


def section_final_test(test_parquet_path: Path | None, out_dir: Path, tables_dir: Path):
    """Generate test_summary.tex and 2 figures from the held-out test parquet."""
    if test_parquet_path is None:
        print("E. Final test skipped (no --test-parquet)")
        return
    if not test_parquet_path.exists():
        print(f"E. Final test skipped ({test_parquet_path} not found)")
        return
    print("E. Final test")
    test_df = pd.read_parquet(test_parquet_path)

    TEST_TRAIN_N = 95
    slice_at = test_df[test_df["p_train_n"] == TEST_TRAIN_N]

    adapter_labels = {
        "GMMAdapter":       "GMM K=1 diag",
        "CosineAdapter":    "Cosine",
        "KNNAdapter":       "kNN k=5",
        "PrototypeAdapter": "Prototype",
        "SmallAEAdapter":   "SmallAE lat=8 ep=100",
    }
    order = (slice_at.groupby("p_adapter")["m_eer"].mean()
             .sort_values().index.tolist())
    adapters = [a for a in order if a in adapter_labels]

    tables_dir.mkdir(parents=True, exist_ok=True)
    metric_labels = {"m_eer": "EER", "m_auprc": "AUPRC", "m_acc_at_far5": "ACC@FAR=5\\%"}
    metrics = list(metric_labels)
    agg = slice_at.groupby("p_adapter")[metrics].agg(["mean", "std"])
    rows = []
    for a in adapters:
        cells = " & ".join(
            f"{agg.loc[a, (m, 'mean')]:.3f} $\\pm$ {agg.loc[a, (m, 'std')]:.3f}"
            for m in metrics
        )
        rows.append(f"    {adapter_labels[a]} & {cells} \\\\")
    tex = "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Final-test metrics on the 5 held-out test words"
        f" at \\texttt{{train\\_n}}={TEST_TRAIN_N}"
        " (mean $\\pm$ std across 5 words $\\times$ 10 trials).}",
        "  \\label{tab:test_summary}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{l{'r' * len(metrics)}}}",
        "    \\toprule",
        "    Adapter & " + " & ".join(metric_labels[m] for m in metrics) + " \\\\",
        "    \\midrule",
        "\n".join(rows),
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "\\end{table}",
    ])
    (tables_dir / "test_summary.tex").write_text(tex + "\n")
    print(f"  saved {tables_dir / 'test_summary.tex'}")

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    tick_labels = [adapter_labels[a] for a in adapters]

    means, margins = [], []
    for a in adapters:
        vals = slice_at.loc[slice_at["p_adapter"] == a, "m_eer"].dropna()
        n = len(vals)
        sem = vals.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        means.append(vals.mean())
        margins.append(t_crit * sem)
    fig, ax = plt.subplots()
    ax.bar(tick_labels, means, yerr=margins, capsize=6, color=palette[:len(adapters)])
    ax.set_ylabel("EER")
    ax.set_title(f"Final-test EER (5 held-out words, train_n={TEST_TRAIN_N}, 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(out_dir, "test_eer_ci")

    fig, ax = plt.subplots()
    for a, color in zip(adapters, palette):
        sub = test_df[test_df["p_adapter"] == a]
        agg_tn = sub.groupby("p_train_n")["m_eer"].agg(["mean", "std", "count"]).reset_index()
        agg_tn = agg_tn.sort_values("p_train_n")
        ci = 1.96 * agg_tn["std"] / np.sqrt(agg_tn["count"])
        ax.plot(agg_tn["p_train_n"], agg_tn["mean"], marker="o",
                label=adapter_labels[a], color=color)
        ax.fill_between(agg_tn["p_train_n"],
                        agg_tn["mean"] - ci, agg_tn["mean"] + ci,
                        alpha=0.15, color=color)
    ax.set_xlabel("Enrollment size (train_n)")
    ax.set_ylabel("EER")
    ax.set_title("Final-test EER vs enrollment budget (5 held-out words)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(out_dir, "test_eer_vs_train_n")

    if {"GMMAdapter", "SmallAEAdapter"}.issubset(set(adapters)):
        idx = ["p_trial", "p_target_class"]
        gmm_eer = slice_at[slice_at["p_adapter"] == "GMMAdapter"].set_index(idx)["m_eer"]
        ae_eer = slice_at[slice_at["p_adapter"] == "SmallAEAdapter"].set_index(idx)["m_eer"]
        paired = pd.concat([gmm_eer.rename("gmm"), ae_eer.rename("ae")], axis=1).dropna()
        diff = paired["ae"] - paired["gmm"]
        t, p = stats.ttest_rel(paired["ae"], paired["gmm"])
        d = diff.mean() / diff.std(ddof=1)
        print(f"  paired t-test (AE - GMM) at train_n={TEST_TRAIN_N}: "
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
    print(f"Loaded {len(df)} rows from {args.parquet}")
    print(f"Writing PDFs to {out_dir}")
    print(f"Writing tables to {tables_dir}\n")

    section_hyperparam(df, out_dir)
    section_compare(df, out_dir)
    section_confidence(df, out_dir)
    section_cost(df, out_dir)
    section_tables(df, tables_dir, dataset)
    section_final_test(args.test_parquet, out_dir, tables_dir)
    print_headline_table(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
