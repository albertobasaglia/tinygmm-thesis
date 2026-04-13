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
    plot_far_recall,
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
    GMM_BEST       = {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "diag"}
    KNN_BEST       = {"p_adapter": "KNNAdapter", "p_k": 5}
    AE_BEST        = {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                      "p_dropout_p": 0.2, "p_threshold_mode": "train"}

    FIXED_TRAIN_N = 45  # mid-budget comparison point (few-shot regime)

    best_lines = [
        ("GMM K=1 diag", GMM_BEST),
        ("kNN k=5",      KNN_BEST),
        ("SmallAE",      AE_BEST),
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

    # A4. AE threshold-mode ablation (pin dropout_p=0.2)
    threshold_lines = [
        ("AE threshold=val",   {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                                "p_dropout_p": 0.2, "p_threshold_mode": "val"}),
        ("AE threshold=train", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                                "p_dropout_p": 0.2, "p_threshold_mode": "train"}),
    ]
    plot_eer(df, lines=threshold_lines)
    plt.title("AE threshold-mode ablation (dropout_p=0.2)")
    save("ae_threshold_mode_eer")

    # A5. AE dropout ablation (pin threshold_mode=train)
    dropout_lines = [
        ("AE no-dropout",    {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                              "p_dropout_p": 0.0, "p_threshold_mode": "train"}),
        ("AE dropout_p=0.2", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 30,
                              "p_dropout_p": 0.2, "p_threshold_mode": "train"}),
    ]
    plot_eer(df, lines=dropout_lines)
    plt.title("AE dropout ablation (threshold_mode=train)")
    save("ae_dropout_eer")

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

    sub = df[df["p_train_n"] == FIXED_TRAIN_N]

    fig, ax = plt.subplots()
    _ci_plot(ax, "m_eer", sub, best_lines, "EER (lower is better)")
    ax.set_title(f"EER with 95% CI (train_n={FIXED_TRAIN_N})")
    fig.tight_layout()
    save("ci_eer")

    fig, ax = plt.subplots()
    _ci_plot(ax, "m_acc_at_far5", sub, best_lines, "ACC @ FAR=5% (higher is better)")
    ax.set_title(f"ACC @ FAR=5% with 95% CI (train_n={FIXED_TRAIN_N})")
    fig.tight_layout()
    save("ci_acc_at_far5")

    # ==================================================================
    # D. AE CONVERGENCE
    # ==================================================================
    print("D. AE convergence")

    # Only plot ep=200: shorter runs would collapse onto the left edge of
    # the shared x-axis and overlap, making the curves unreadable.
    loss_lines = [
        ("SmallAE ep=200", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 200}),
    ]
    if "m_val_loss_1" in df.columns and not _filter(df, loss_lines[0][1]).empty:
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

    fig, ax = plt.subplots(figsize=(FULL_W, H + 0.5))
    for i, (label, where) in enumerate(best_lines):
        means = [
            _filter(sub, where).groupby("p_target_class")["m_acc_at_far5"].mean().get(w, float("nan"))
            for w in words
        ]
        offset = (i - len(best_lines) / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=30, ha="right")
    ax.set_ylabel("ACC @ FAR=5%")
    ax.set_title(f"ACC @ FAR=5% by target word (train_n={FIXED_TRAIN_N})")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save("acc_at_far5_per_word")

    # ==================================================================
    # G. FINAL TEST (held-out TEST_WORDS)
    # ==================================================================
    test_path = ROOT / "results" / "test.parquet"
    if test_path.exists():
        print("G. Final test")
        test_df = pd.read_parquet(test_path)
        print(f"  loaded {len(test_df)} rows | adapters: {test_df['p_adapter'].unique().tolist()}")

        # G1. Summary table (LaTeX)
        tables_dir = ROOT / "tinygmm-tex" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        metric_labels = {
            "m_eer": "EER",
            "m_auc": "AUC",
            "m_auprc": "AUPRC",
            "m_f1": "F1",
            "m_acc_at_far5": "ACC@FAR=5\\%",
        }
        metrics = [m for m in metric_labels if m in test_df.columns]
        adapter_labels = {"GMMAdapter": "GMM K=1 diag", "SmallAEAdapter": "SmallAE ep=200"}
        agg = test_df.groupby("p_adapter")[metrics].agg(["mean", "std"])
        col_header = " & ".join(metric_labels[m] for m in metrics)
        rows = []
        for adapter_key, label in adapter_labels.items():
            if adapter_key not in agg.index:
                continue
            cells = " & ".join(
                f"{agg.loc[adapter_key, (m, 'mean')]:.3f} $\\pm$ {agg.loc[adapter_key, (m, 'std')]:.3f}"
                for m in metrics
            )
            rows.append(f"    {label} & {cells} \\\\")
        n_cols = 1 + len(metrics)
        col_spec = "l" + "r" * len(metrics)
        tex = "\n".join([
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{Final-test metrics on the 5 held-out words"
            " (mean $\\pm$ std across 5 words $\\times$ 10 trials).}",
            "  \\label{tab:test_summary}",
            "  \\resizebox{\\textwidth}{!}{%",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            "    \\toprule",
            f"    Adapter & {col_header} \\\\",
            "    \\midrule",
            "\n".join(rows),
            "    \\bottomrule",
            "  \\end{tabular}%",
            "  }",
            "\\end{table}",
        ])
        (tables_dir / "test_summary.tex").write_text(tex + "\n")
        print("  saved tables/test_summary.tex")

        # G2. EER bar chart with 95% CI
        adapters = sorted(test_df["p_adapter"].unique())
        means, margins = [], []
        for a in adapters:
            vals = test_df.loc[test_df["p_adapter"] == a, "m_eer"].dropna()
            n = len(vals)
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            means.append(mean)
            margins.append(t_crit * sem)
        fig, ax = plt.subplots()
        ax.bar(adapters, means, yerr=margins, capsize=6, color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("EER")
        ax.set_title("Final test EER (5 held-out words, 95% CI)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save("test_eer_ci")

        # G3. Paired t-test GMM vs SmallAE (stdout only)
        if set(adapters) >= {"GMMAdapter", "SmallAEAdapter"}:
            idx = ["p_trial", "p_target_class"]
            gmm_eer = test_df[test_df["p_adapter"] == "GMMAdapter"].set_index(idx)["m_eer"]
            ae_eer = test_df[test_df["p_adapter"] == "SmallAEAdapter"].set_index(idx)["m_eer"]
            paired = pd.concat([gmm_eer.rename("gmm"), ae_eer.rename("ae")], axis=1).dropna()
            diff = paired["ae"] - paired["gmm"]
            t, p = stats.ttest_rel(paired["ae"], paired["gmm"])
            cohen_d = diff.mean() / diff.std(ddof=1)
            print(
                f"  paired t-test (AE - GMM): n={len(paired)}  "
                f"mean={diff.mean():+.4f}  d={cohen_d:+.3f}  t={t:+.3f}  p={p:.4g}"
            )

        # G4. EER boxplot
        data = [test_df.loc[test_df["p_adapter"] == a, "m_eer"].dropna().values for a in adapters]
        fig, ax = plt.subplots()
        bp = ax.boxplot(
            data, tick_labels=adapters, patch_artist=True, showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("EER")
        ax.set_title("Final test EER distribution (5 words x 10 trials)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save("test_eer_box")

        # G5. FAR vs Recall scatter
        test_lines = [
            ("GMM K=1 diag",   {"p_adapter": "GMMAdapter"}),
            ("SmallAE ep=200", {"p_adapter": "SmallAEAdapter"}),
        ]
        plot_far_recall(test_df, lines=test_lines)
        save("test_far_recall")
    else:
        print(f"G. Final test skipped ({test_path} not found)")

    print("\nDone.")


if __name__ == "__main__":
    main()
