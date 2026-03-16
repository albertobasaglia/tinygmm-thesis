"""
Comparison framework for one-class adapters (autoencoder, GMM, ...).

Usage:
    cd src
    python -m compare

Edit CHECKPOINTS and make_configs() to control what gets compared.
Edit the PLOTS section to control what gets visualised.
"""

from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

from lib.models import SpeechExtractorModule
from lib.data import get_spectrograms

from .adapters import AutoencoderAdapter, GMMAdapter
from .metrics import evaluate
from .sweep import sweep
from .plots import (
    plot_lines,
    plot_sweep,
    plot_far_recall,
    plot_eer,
    plot_auc_auprc,
    plot_precision_recall_bar,
    plot_f1,
    plot_eer_by_dim,
    plot_eer_train_n_by_dim,
)


def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    ROOT = Path(__file__).parent.parent.parent   # repo root

    # =================================================================
    # CHECKPOINTS
    #
    # List of (ckpt_path, embedding_dim) pairs.  Each entry is one
    # Stage-1 feature extractor trained with a specific embedding size.
    # The sweep below is run independently for every checkpoint and
    # results are tagged with embedding_dim so they can be plotted together.
    #
    # Train additional checkpoints with:
    #   python train_speech_extractor.py --embedding_dim 16
    # then copy the best .ckpt here and add an entry.
    # =================================================================
    CHECKPOINTS = [
        (ROOT / "best.ckpt", 32),
        # (ROOT / "best_dim16.ckpt", 16),
        # (ROOT / "best_dim8.ckpt",   8),
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
    # injected automatically from the checkpoint's embedding_dim below.
    # =================================================================
    TRAIN_N = 64

    train_n = [10, 15, 20, 25, 30, 40, 50, 100]

    def make_configs(embedding_dim: int) -> list:
        return [
            *sweep(AutoencoderAdapter, {
                "train_n": train_n,
                "epochs": [100],
                "device": [DEVICE],
                "input_dim": [embedding_dim],
            }),
            *sweep(GMMAdapter, {
                "train_n": train_n,
                "n_components": [1, 2],
                "covariance_type": ["full", "diag"],
            }),
        ]

    # --- Extract spectrograms once (shared across all checkpoints) ---
    print("Loading spectrograms...")
    data_dir = str(ROOT / "data")
    specs_train = get_spectrograms(data_dir, target_class="yes", n=TRAIN_N)
    specs_yes   = get_spectrograms(data_dir, target_class="yes", n=TEST_N, subset="testing")
    specs_no    = get_spectrograms(data_dir, target_class="no",  n=TEST_N, subset="testing")

    # --- Loop over checkpoints ---
    rows = []
    for ckpt_path, embedding_dim in CHECKPOINTS:
        print(f"\n=== embedding_dim={embedding_dim} ({ckpt_path}) ===")
        extractor = SpeechExtractorModule.load_from_checkpoint(ckpt_path)
        extractor.to(DEVICE).eval()

        with torch.no_grad():
            train_emb   = extractor(specs_train.to(DEVICE), return_embedding=True).cpu().numpy()
            test_target = extractor(specs_yes.to(DEVICE),   return_embedding=True).cpu().numpy()
            test_other  = extractor(specs_no.to(DEVICE),    return_embedding=True).cpu().numpy()

        for name, cls, kwargs in make_configs(embedding_dim):
            print(f"  {name}...")
            adapter = cls(**kwargs)
            adapter.fit(train_emb)
            rows.append({
                "embedding_dim": embedding_dim,
                "adapter": cls.__name__,
                **kwargs,
                **evaluate(adapter, test_target, test_other),
            })

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    # =================================================================
    # PLOTS
    #
    # --- Per-adapter comparisons (single embedding_dim) ---
    # plot_lines:               generic y vs x, one line per config
    # plot_far_recall:          operating point scatter (FAR vs Recall)
    # plot_eer:                 EER vs train_n (threshold-free)
    # plot_auc_auprc:           AUC-ROC vs AUPRC on one chart
    # plot_precision_recall_bar precision/recall bars at a fixed train_n
    # plot_f1:                  F1 vs train_n
    #
    # --- Embedding-dim ablation (multiple checkpoints) ---
    # plot_eer_by_dim:          EER vs embedding_dim at fixed train_n
    # plot_eer_train_n_by_dim:  EER vs train_n, one line per embedding_dim
    #
    # Each line is (label, filter_dict) where filter_dict matches columns.
    # =================================================================
    lines = [
        ("AE",           {"adapter": "AutoencoderAdapter", "epochs": 100}),
        ("GMM diag n=1", {"adapter": "GMMAdapter", "n_components": 1, "covariance_type": "diag"}),
        ("GMM diag n=2", {"adapter": "GMMAdapter", "n_components": 2, "covariance_type": "diag"}),
        ("GMM full n=1", {"adapter": "GMMAdapter", "n_components": 1, "covariance_type": "full"}),
        ("GMM full n=2", {"adapter": "GMMAdapter", "n_components": 2, "covariance_type": "full"}),
    ]
    lines_good = lines[:3]  # exclude degenerate full-cov configs for cleaner plots

    # Classic scalar metric lines
    # plot_lines(df, x="train_n", y="auc", lines=lines_good)

    # Operating point scatter — exposes degenerate full-cov configs
    # plot_far_recall(df, lines=lines)

    # Threshold-free comparison: EER (lower = better)
    # plot_eer(df, lines=lines_good)

    # AUC-ROC vs AUPRC — flags cases where AUC flatters a config
    # plot_auc_auprc(df, lines=lines_good)

    # Precision & Recall at one representative budget
    # plot_precision_recall_bar(df, train_n=50, lines=lines)

    # F1 penalises degenerate configs, clean single-number comparison
    # plot_f1(df, lines=lines_good)

    # --- Embedding-dim ablation (uncomment when multiple checkpoints are loaded) ---
    # EER vs embedding_dim at a fixed enrollment budget
    # plot_eer_by_dim(df, lines=lines_good, fixed_train_n=50)

    # EER vs train_n with one line per embedding_dim, for a single adapter config
    # plot_eer_train_n_by_dim(df, where={"adapter": "GMMAdapter", "covariance_type": "diag", "n_components": 1})

    plt.show()

    return df


if __name__ == "__main__":
    main()
