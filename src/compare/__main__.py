"""
Comparison framework for one-class adapters (autoencoder, GMM, ...).

Usage:
    python -m src.compare

Edit CHECKPOINTS and make_configs() to control what gets compared.
Results are saved as a Parquet file in results/.
"""

from pathlib import Path

import numpy as np
import torch
import pandas as pd

from lib.models import SpeechExtractorModule
from lib.data import get_spectrograms

from .adapters import AutoencoderAdapter, GMMAdapter, KNNAdapter
from .metrics import evaluate
from .sweep import sweep


def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500
    N_TRIALS = 10
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
        (ROOT / "best_32.ckpt", 32),
        (ROOT / "best_16.ckpt", 16),
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
    train_n = [10, 15, 20, 25, 30]

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
            *sweep(KNNAdapter, {
                "train_n": train_n,
                "k": [1, 3, 5],
            }),
        ]

    # --- Extract spectrograms once (shared across all checkpoints) ---
    print("Loading spectrograms...")
    data_dir = str(ROOT / "data")
    specs_train = get_spectrograms(data_dir, target_class="yes", n=max(train_n))
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

        for trial in range(N_TRIALS):
            rng = np.random.default_rng(seed=trial)
            shuffled_emb = rng.permutation(train_emb)

            for name, cls, kwargs in make_configs(embedding_dim):
                if trial == 0:
                    print(f"  {name}...")
                adapter = cls(**kwargs)
                adapter.fit(shuffled_emb)
                rows.append({
                    "p_trial": trial,
                    "p_embedding_dim": embedding_dim,
                    "p_adapter": cls.__name__,
                    **{f"p_{k}": v for k, v in kwargs.items()},
                    **evaluate(adapter, test_target, test_other),
                })

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    # --- Save results ---
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "sweep.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    return df


if __name__ == "__main__":
    main()
