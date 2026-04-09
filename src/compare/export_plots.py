"""
Export thesis plots to PDF.

Usage:
    python -m src.compare.export_plots

Reads results/sweep.parquet and writes PDFs to tinygmm-tex/figures/.

Plot structure (matches thesis narrative):
  A. Hyperparameter selection  (appendix)
  B. Main comparison           (best-of-each: GMM vs kNN vs AE)
  C. Statistical confidence    (CI error bars)
  D. AE convergence            (loss curves)
  E. Computational cost        (FLOPs vs EER, Pareto frontiers)
  F. Per-word robustness       (EER per target word)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from .plots import (
    _filter, _agg, _plot_line,
    plot_eer,
    plot_lines,
    plot_gmm_components,
    plot_gmm_diag_vs_full,
    plot_loss_curves,
    plot_pareto,
)

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "tinygmm-tex" / "figures"


# --- Thesis rcParams ---
# Figure width: ~13 cm text width on A4 = 5.12 in
FULL_W = 5.12
HALF_W = 2.5
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "results" / "sweep.parquet")
    print(f"Loaded {len(df)} rows from sweep.parquet")
    print(f"Writing PDFs to {OUT}\n")

    # ------------------------------------------------------------------
    # Best-of-each configs (from sweep results)
    # ------------------------------------------------------------------
    GMM_BEST = {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "diag"}
    KNN_BEST = {"p_adapter": "KNNAdapter", "p_k": 5}
    AE_BEST  = {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30}

    FIXED_TRAIN_N = 95  # high-budget comparison point

    best_lines = [
        ("GMM K=1 diag", GMM_BEST),
        ("kNN k=5",      KNN_BEST),
        ("SmallAE ep=30", AE_BEST),
    ]

    # ==================================================================
    # A. HYPERPARAMETER SELECTION (appendix)
    # ==================================================================
    print("A. Hyperparameter selection")

    # A1. GMM covariance type comparison
    plot_gmm_diag_vs_full(df, y="m_eer")
    save("gmm_cov_eer")

    # A2. GMM number of components
    plot_gmm_components(df, y="m_eer", fixed_train_n=FIXED_TRAIN_N)
    save("gmm_components_eer")

    # A3. kNN: EER vs k at fixed train_n
    knn_sub = _filter(df, {"p_adapter": "KNNAdapter"})
    knn_sub = knn_sub[knn_sub["p_train_n"] == FIXED_TRAIN_N]
    fig, ax = plt.subplots()
    agg = knn_sub.groupby("p_k")["m_eer"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.sort_values("p_k")
    ax.bar(agg["p_k"], agg["mean"], yerr=agg["std"], capsize=3)
    ax.set_xlabel("k")
    ax.set_ylabel("EER")
    ax.set_title(f"kNN: EER vs k (train_n={FIXED_TRAIN_N})")
    ax.set_xticks(agg["p_k"].astype(int))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save("knn_k_selection")

    # ==================================================================
    # B. MAIN COMPARISON (best-of-each)
    # ==================================================================
    print("B. Main comparison")

    plot_eer(df, lines=best_lines)
    save("compare_eer")

    plot_lines(df, x="p_train_n", y="m_auc", lines=best_lines)
    save("compare_auc")

    plot_lines(df, x="p_train_n", y="m_acc_at_far5", lines=best_lines)
    plt.ylabel("ACC @ FAR=5%")
    plt.title("Accuracy at FAR=5% vs enrollment budget")
    save("compare_acc_at_far")

    # ==================================================================
    # C. STATISTICAL CONFIDENCE
    # ==================================================================
    print("C. Statistical confidence")

    sub = df[df["p_train_n"] == FIXED_TRAIN_N]
    fig, ax = plt.subplots()
    for i, (label, where) in enumerate(best_lines):
        vals = _filter(sub, where)["m_eer"]
        if vals.empty:
            continue
        mean = vals.mean()
        n = len(vals)
        sem = vals.std() / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci = t_crit * sem
        ax.errorbar(mean, i, xerr=ci, fmt="o", capsize=5, markersize=6)
        ax.text(mean + ci + 0.005, i, f"{mean:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(best_lines)))
    ax.set_yticklabels([c[0] for c in best_lines])
    ax.set_xlabel("EER (lower is better)")
    ax.set_title(f"EER with 95% CI (train_n={FIXED_TRAIN_N})")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save("ci_eer")

    # ==================================================================
    # D. AE CONVERGENCE
    # ==================================================================
    print("D. AE convergence")

    loss_lines = [
        ("ep=10", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 10}),
        ("ep=20", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 20}),
        ("ep=30", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30}),
    ]
    if "m_loss_1" in df.columns and not _filter(df, {"p_adapter": "SmallAEAdapter"}).empty:
        plot_loss_curves(df, lines=loss_lines)
        save("ae_loss_curves")

    # ==================================================================
    # E. COMPUTATIONAL COST
    # ==================================================================
    print("E. Computational cost")

    # E1. Inference FLOPs bar chart (best-of-each, at fixed train_n)
    fig, ax = plt.subplots()
    labels, flops = [], []
    sub_fixed = df[df["p_train_n"] == FIXED_TRAIN_N]
    for label, where in best_lines:
        vals = _filter(sub_fixed, where)["m_inference_flops"].dropna()
        if vals.empty:
            continue
        labels.append(label)
        flops.append(vals.mean())
    bars = ax.bar(labels, flops)
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("Inference FLOPs")
    ax.set_title(f"Inference cost per sample (train\_n={FIXED_TRAIN_N})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save("inference_flops_bar")

    # E1b. Inference FLOPs vs train_n (shows kNN scaling)
    fig, ax = plt.subplots()
    for label, where in best_lines:
        sub = _filter(df, where)
        agg = sub.groupby("p_train_n")["m_inference_flops"].mean().reset_index()
        agg = agg.sort_values("p_train_n")
        ax.plot(agg["p_train_n"], agg["m_inference_flops"], marker="o", label=label)
    ax.set_xlabel("Enrollment size (train\_n)")
    ax.set_ylabel("Inference FLOPs per sample")
    ax.set_title("Inference cost vs enrollment size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save("inference_flops_vs_train_n")

    # E2. Training FLOPs vs EER
    fig, ax = plt.subplots()
    for label, where in best_lines:
        sub = _filter(df, where)
        if sub.empty:
            continue
        agg = _agg(sub, "m_training_flops", "m_eer")
        ax.plot(agg["m_training_flops"], agg["mean"], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs")
    ax.set_ylabel("EER")
    ax.set_title("EER vs training cost")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save("flops_training_eer")

    # E3. Pareto frontiers
    pareto_lines = [
        ("SmallAE", {"p_adapter": "SmallAEAdapter"}),
        ("GMM",     {"p_adapter": "GMMAdapter"}),
        ("kNN",     {"p_adapter": "KNNAdapter"}),
    ]
    plot_pareto(df, lines=pareto_lines, x="m_training_flops")
    save("pareto_training")

    plot_pareto(df, lines=pareto_lines, x="m_inference_flops")
    save("pareto_inference")

    # ==================================================================
    # F. PER-WORD ROBUSTNESS
    # ==================================================================
    print("F. Per-word robustness")

    sub = df[df["p_train_n"] == FIXED_TRAIN_N]
    words = sorted(df["p_target_class"].unique())
    x = np.arange(len(words))
    width = 0.8 / len(best_lines)

    fig, ax = plt.subplots(figsize=(FULL_W, H + 0.5))
    for i, (label, where) in enumerate(best_lines):
        means = [
            _filter(sub, where).groupby("p_target_class")["m_eer"].mean().get(w, float("nan"))
            for w in words
        ]
        offset = (i - len(best_lines) / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=30, ha="right")
    ax.set_ylabel("EER")
    ax.set_title(f"EER by target word (train_n={FIXED_TRAIN_N})")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save("eer_per_word")

    print("\nDone.")


if __name__ == "__main__":
    main()
