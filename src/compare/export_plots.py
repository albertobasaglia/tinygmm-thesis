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
  E. Final test on held-out words (test_summary table + 2 figures) — requires --test-parquet
"""

import argparse
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

FIXED_TRAIN_N = 25


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask for Pareto-optimal points (lower x, lower y preferred)."""
    is_pareto = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominated = ((x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i])))
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False
    return is_pareto


def _save(out_dir: Path, name: str):
    plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  saved {name}.pdf")


def section_hyperparam(df: pd.DataFrame, out_dir: Path):
    print("A. Hyperparameter selection")

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

    knn_sub = _filter(df, {"p_adapter": "KNNAdapter"})
    knn_sub = knn_sub[knn_sub["p_train_n"] == FIXED_TRAIN_N]
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
    print("D. Computational cost")

    fig, ax = plt.subplots()
    labels, flops = [], []
    sub_fixed = df[df["p_train_n"] == FIXED_TRAIN_N]
    for label, where in BEST_LINES:
        vals = _filter(sub_fixed, where)["m_inference_flops"].dropna()
        if vals.empty:
            continue
        labels.append(label)
        flops.append(vals.mean())
    bars = ax.bar(labels, flops)
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("Inference FLOPs")
    ax.set_title(f"Inference cost per sample (train_n={FIXED_TRAIN_N})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(out_dir, "inference_flops_bar")

    fig, ax = plt.subplots()
    for label, where in BEST_LINES:
        sub = _filter(df, where)
        agg = sub.groupby("p_train_n")["m_inference_flops"].mean().reset_index().sort_values("p_train_n")
        ax.plot(agg["p_train_n"], agg["m_inference_flops"], marker="o", label=label)
    ax.set_xlabel("Enrollment size (train_n)")
    ax.set_ylabel("Inference FLOPs per sample")
    ax.set_title("Inference cost vs enrollment size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(out_dir, "inference_flops_vs_train_n")

    fig, ax = plt.subplots()
    for label, where in PARETO_LINES:
        subset = _filter(df, where)
        if subset.empty:
            continue
        agg = subset.groupby("m_inference_flops")["m_eer"].mean().reset_index()
        xs, ys = agg["m_inference_flops"].values, agg["m_eer"].values
        color = ax._get_lines.get_next_color()
        ax.scatter(xs, ys, alpha=0.4, s=20, color=color)
        pareto = _pareto_mask(xs, ys)
        if pareto.any():
            px, py = xs[pareto], ys[pareto]
            order = np.argsort(px)
            ax.scatter(px, py, s=80, color=color, label=label, zorder=3)
            ax.plot(px[order], py[order], color=color, linewidth=1.5, alpha=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Inference FLOPs")
    ax.set_ylabel("EER (lower is better)")
    ax.set_title("Pareto Frontier: EER vs Inference FLOPs")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    _save(out_dir, "pareto_inference")


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
                        help="Output directory for PDFs. Defaults to figures/<derived-name>.")
    parser.add_argument("--test-parquet", type=Path, default=None,
                        help="Optional held-out test parquet. Enables section E (final test).")
    args = parser.parse_args()

    out_dir = args.out if args.out is not None else _derive_out_dir(args.parquet)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir.parent / "tables"

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")
    print(f"Writing PDFs to {out_dir}\n")

    section_hyperparam(df, out_dir)
    section_compare(df, out_dir)
    section_confidence(df, out_dir)
    section_cost(df, out_dir)
    section_final_test(args.test_parquet, out_dir, tables_dir)
    print_headline_table(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
