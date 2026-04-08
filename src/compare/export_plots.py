"""
Export thesis plots to PDF.

Usage:
    python -m src.compare.export_plots

Reads results/sweep.parquet and writes PDFs to tinygmm-tex/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plots import (
    _filter, _agg,
    plot_eer,
    plot_f1,
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
# Use half/full widths and 10pt body font.
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
    # Line configs — edit these to control what appears in each plot
    # ------------------------------------------------------------------
    gmm_lines = [
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

    compare_lines = [
        ("GMM K=1 diag", {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "diag"}),
        ("GMM K=2 diag", {"p_adapter": "GMMAdapter", "p_n_components": 2, "p_covariance_type": "diag"}),
        ("GMM K=3 diag", {"p_adapter": "GMMAdapter", "p_n_components": 3, "p_covariance_type": "diag"}),
        ("kNN k=1",      {"p_adapter": "KNNAdapter", "p_k": 1}),
    ]

    fair_lines = [
        ("SmallAE ep=10",  {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 10}),
        ("SmallAE ep=50",  {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 50}),
        ("GMM K=1 diag",   {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "diag"}),
        ("GMM K=2 diag",   {"p_adapter": "GMMAdapter", "p_n_components": 2, "p_covariance_type": "diag"}),
        ("GMM K=3 diag",   {"p_adapter": "GMMAdapter", "p_n_components": 3, "p_covariance_type": "diag"}),
        ("GMM K=1 sph",    {"p_adapter": "GMMAdapter", "p_n_components": 1, "p_covariance_type": "spherical"}),
        ("kNN k=1",        {"p_adapter": "KNNAdapter", "p_k": 1}),
    ]

    pareto_lines = [
        ("SmallAE", {"p_adapter": "SmallAEAdapter"}),
        ("GMM",     {"p_adapter": "GMMAdapter"}),
        ("kNN",     {"p_adapter": "KNNAdapter"}),
    ]

    loss_lines = [
        ("ep=10",  {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 10}),
        ("ep=50",  {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 50}),
        ("ep=100", {"p_adapter": "SmallAEAdapter", "p_latent_dim": 4, "p_epochs": 100}),
    ]

    # ------------------------------------------------------------------
    # GMM internal comparison
    # ------------------------------------------------------------------
    plot_gmm_diag_vs_full(df, y="m_eer")
    save("gmm_cov_eer")

    plot_gmm_diag_vs_full(df, y="m_auc")
    save("gmm_cov_auc")

    plot_gmm_components(df, y="m_eer", fixed_train_n=45)
    save("gmm_components_eer")

    plot_gmm_components(df, y="m_auc", fixed_train_n=45)
    save("gmm_components_auc")

    # ------------------------------------------------------------------
    # Cross-adapter comparison
    # ------------------------------------------------------------------
    plot_eer(df, lines=compare_lines)
    save("compare_eer")

    plot_lines(df, x="p_train_n", y="m_auc", lines=compare_lines)
    save("compare_auc")

    plot_f1(df, lines=compare_lines)
    save("compare_f1")

    # ------------------------------------------------------------------
    # FLOPs vs EER (training & inference)
    # ------------------------------------------------------------------
    import matplotlib.cm as cm
    colors = cm.tab20.colors

    for dim in sorted(df["p_embedding_dim"].unique()):
        fig, ax = plt.subplots()
        for ci, (label, where) in enumerate(fair_lines):
            sub = _filter(df[df["p_embedding_dim"] == dim], where)
            if sub.empty:
                continue
            agg = _agg(sub, "m_training_flops", "m_eer")
            ax.plot(agg["m_training_flops"], agg["mean"], marker="o",
                    label=label, color=colors[ci % len(colors)])
        ax.set_xscale("log")
        ax.set_xlabel("Training FLOPs")
        ax.set_ylabel("EER")
        ax.set_title(f"EER vs Training FLOPs (dim={dim})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save(f"flops_training_eer_dim{dim}")

    for dim in sorted(df["p_embedding_dim"].unique()):
        fig, ax = plt.subplots()
        for ci, (label, where) in enumerate(fair_lines):
            sub = _filter(df[df["p_embedding_dim"] == dim], where)
            if sub.empty:
                continue
            agg = _agg(sub, "m_inference_flops", "m_eer")
            ax.plot(agg["m_inference_flops"], agg["mean"], marker="o",
                    label=label, color=colors[ci % len(colors)])
        ax.set_xscale("log")
        ax.set_xlabel("Inference FLOPs")
        ax.set_ylabel("EER")
        ax.set_title(f"EER vs Inference FLOPs (dim={dim})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save(f"flops_inference_eer_dim{dim}")

    # ------------------------------------------------------------------
    # Pareto frontiers
    # ------------------------------------------------------------------
    plot_pareto(df, lines=pareto_lines, x="m_training_flops")
    save("pareto_training")

    plot_pareto(df, lines=pareto_lines, x="m_inference_flops")
    save("pareto_inference")

    # ------------------------------------------------------------------
    # AE loss convergence (only if SmallAE rows exist)
    # ------------------------------------------------------------------
    if "m_loss_1" in df.columns and not _filter(df, {"p_adapter": "SmallAEAdapter"}).empty:
        plot_loss_curves(df, lines=loss_lines)
        save("ae_loss_curves")

    print("\nDone.")


if __name__ == "__main__":
    main()
