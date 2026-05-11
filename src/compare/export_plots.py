"""
Export thesis plots to PDF.

Usage:
    python -m src.compare.export_plots

Reads results/sweep_speech_latest.parquet and writes 12 PDFs to
tinygmm-tex/figures/.  Also prints headline numbers (best-of-each adapter at
train_n=45) for the inline LaTeX table in results.tex.

Plot structure (matches the results chapter):
  A. Hyperparameter selection  (covariance, K, k, AE dropout)
  B. Main adapter comparison   (best-of-each)
  C. Statistical confidence    (95% CI bar charts)
  D. Computational cost        (inference FLOPs + Pareto)
  E. Final test on held-out words (test_summary table + 2 figures)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from .plots import (
    _filter, _plot_line,
    plot_eer,
    plot_lines,
    plot_gmm_components,
    plot_gmm_diag_vs_full,
    plot_pareto,
)

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "tinygmm-tex" / "figures"

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


def save(name: str):
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  saved {name}.pdf")


# Headline configurations
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

PARETO_LINES = [
    ("SmallAE",   {"p_adapter": "SmallAEAdapter"}),
    ("GMM",       {"p_adapter": "GMMAdapter"}),
    ("kNN",       {"p_adapter": "KNNAdapter"}),
    ("Cosine",    {"p_adapter": "CosineAdapter"}),
    ("Prototype", {"p_adapter": "PrototypeAdapter"}),
]

FIXED_TRAIN_N = 45


def _ci_plot(ax, metric, sub, lines, xlabel):
    for i, (label, where) in enumerate(lines):
        vals = _filter(sub, where)[metric]
        if vals.empty:
            continue
        mean = vals.mean()
        n = len(vals)
        sem = vals.std() / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci = t_crit * sem
        ax.errorbar(mean, i, xerr=ci, fmt="o", capsize=5, markersize=6)
        ax.text(mean + ci + 0.005, i, f"{mean:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(lines)))
    ax.set_yticklabels([c[0] for c in lines])
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def section_hyperparam(df: pd.DataFrame):
    print("A. Hyperparameter selection")

    plot_gmm_diag_vs_full(df, y="m_eer")
    save("gmm_cov_eer")

    plot_gmm_components(df, y="m_eer", fixed_train_n=FIXED_TRAIN_N)
    save("gmm_components_eer")

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
    save("knn_k_selection")

    dropout_lines = [
        ("AE no-dropout",    {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                              "p_dropout_p": 0.0}),
        ("AE dropout_p=0.2", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                              "p_dropout_p": 0.2}),
    ]
    plot_eer(df, lines=dropout_lines)
    plt.title("AE dropout ablation")
    save("ae_dropout_eer")


def section_compare(df: pd.DataFrame):
    print("B. Main comparison")

    plot_eer(df, lines=BEST_LINES)
    save("compare_eer")

    plot_lines(df, x="p_train_n", y="m_auc", lines=BEST_LINES)
    save("compare_auc")

    plot_lines(df, x="p_train_n", y="m_acc_at_far5", lines=BEST_LINES)
    plt.ylabel("ACC @ FAR=5%")
    plt.title("Accuracy at FAR=5% vs enrollment budget")
    save("compare_acc_at_far")


def section_confidence(df: pd.DataFrame):
    print("C. Statistical confidence")
    sub = df[df["p_train_n"] == FIXED_TRAIN_N]

    fig, ax = plt.subplots()
    _ci_plot(ax, "m_eer", sub, BEST_LINES, "EER (lower is better)")
    ax.set_title(f"EER with 95% CI (train_n={FIXED_TRAIN_N})")
    fig.tight_layout()
    save("ci_eer")

    fig, ax = plt.subplots()
    _ci_plot(ax, "m_acc_at_far5", sub, BEST_LINES, "ACC @ FAR=5% (higher is better)")
    ax.set_title(f"ACC @ FAR=5% with 95% CI (train_n={FIXED_TRAIN_N})")
    fig.tight_layout()
    save("ci_acc_at_far5")


def section_cost(df: pd.DataFrame):
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
    save("inference_flops_bar")

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
    save("inference_flops_vs_train_n")

    plot_pareto(df, lines=PARETO_LINES, x="m_inference_flops")
    save("pareto_inference")


def section_final_test():
    """Generate test_summary.tex and 2 figures from results/test_speech.parquet."""
    test_path = ROOT / "results" / "test_speech.parquet"
    if not test_path.exists():
        print(f"E. Final test skipped ({test_path.name} not found)")
        return
    print("E. Final test")
    test_df = pd.read_parquet(test_path)

    TEST_TRAIN_N = 95
    slice_at = test_df[test_df["p_train_n"] == TEST_TRAIN_N]

    adapter_labels = {
        "GMMAdapter":       "GMM K=1 diag",
        "CosineAdapter":    "Cosine",
        "KNNAdapter":       "kNN k=5",
        "PrototypeAdapter": "Prototype",
        "SmallAEAdapter":   "SmallAE lat=8 ep=100",
    }
    # Order by EER at TEST_TRAIN_N
    order = (slice_at.groupby("p_adapter")["m_eer"].mean()
             .sort_values().index.tolist())
    adapters = [a for a in order if a in adapter_labels]

    # E1. LaTeX summary table
    tables_dir = ROOT / "tinygmm-tex" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metric_labels = {"m_eer": "EER", "m_auc": "AUC",
                     "m_auprc": "AUPRC", "m_acc_at_far5": "ACC@FAR=5\\%"}
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
    print("  saved tables/test_summary.tex")

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    tick_labels = [adapter_labels[a] for a in adapters]

    # E2. EER bar chart with 95% CI at train_n=95
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
    save("test_eer_ci")

    # E3. EER vs train_n
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
    save("test_eer_vs_train_n")

    # Paired t-test: GMM vs SmallAE at TEST_TRAIN_N
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
    """Print headline numbers for the inline LaTeX table in results.tex."""
    print()
    print(f"Headline numbers @ train_n={FIXED_TRAIN_N} (mean over 10 trials x 10 target classes):")
    print()
    sub = df[df["p_train_n"] == FIXED_TRAIN_N]
    fmt = "  {:<14} {:>8} {:>8} {:>8} {:>10} {:>8}"
    print(fmt.format("Adapter", "EER", "AUC", "ACC@5%", "InfFLOPs", "Params"))
    print(fmt.format("-------", "---", "---", "------", "--------", "------"))
    for label, where in BEST_LINES:
        s = _filter(sub, where)
        if s.empty:
            continue
        print(fmt.format(
            label,
            f"{s['m_eer'].mean():.3f}",
            f"{s['m_auc'].mean():.3f}",
            f"{s['m_acc_at_far5'].mean():.3f}",
            f"{s['m_inference_flops'].mean():.0f}",
            f"{s['m_parameters'].mean():.0f}",
        ))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sweep_path = ROOT / "results" / "sweep_speech_latest.parquet"
    df = pd.read_parquet(sweep_path)
    print(f"Loaded {len(df)} rows from {sweep_path.name}")
    print(f"Writing PDFs to {OUT}\n")

    section_hyperparam(df)
    section_compare(df)
    section_confidence(df)
    section_cost(df)
    section_final_test()
    print_headline_table(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
