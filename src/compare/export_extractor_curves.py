"""
Export feature-extractor training curves (train/val loss and accuracy) to PDF.

Reads the Lightning CSVLogger metrics written during extractor training under
logs/<name>/version_*/metrics.csv and emits one two-panel figure per extractor
(loss vs epoch, accuracy vs epoch) for the thesis appendix.

Usage:
    python -m src.compare.export_extractor_curves [--out tinygmm-tex/figures/extractor]

The CSV interleaves train rows (train_* populated, val_* blank) and validation
rows (val_* populated, train_* blank), so metrics are collapsed to one value per
epoch by a NaN-skipping mean before plotting.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

# Match the thesis figure style used by export_plots.py.
FULL_W = 5.12
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# (key, log directory, display name) for each extractor.
EXTRACTORS = [
    ("speech", "speech_extractor", "Speech extractor"),
    ("har", "har_extractor", "HAR extractor"),
]


def _per_epoch(log_dir: Path) -> pd.DataFrame:
    """Collapse the interleaved CSV to one row per epoch (NaN-skipping mean)."""
    versions = sorted(log_dir.glob("version_*"))
    if not versions:
        raise FileNotFoundError(f"no version_* under {log_dir}")
    csv = versions[-1] / "metrics.csv"
    df = pd.read_csv(csv)
    cols = [c for c in ("train_loss", "val_loss", "train_acc", "val_acc") if c in df.columns]
    return df.groupby("epoch")[cols].mean().sort_index()


def _plot_one(epochs, train, val, ylabel: str, out_path: Path, ylim=None):
    """One single-panel figure (train + validation of a single metric)."""
    fig, ax = plt.subplots(figsize=(FULL_W, 3.0))
    ax.plot(epochs, train, label="train", color="#4C72B0")
    ax.plot(epochs, val, label="validation", color="#DD8452")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _plot(curves: pd.DataFrame, key: str, out_dir: Path):
    epochs = curves.index.values
    _plot_one(epochs, curves["train_loss"], curves["val_loss"],
              "Cross-entropy loss", out_dir / f"{key}_loss.pdf")
    _plot_one(epochs, curves["train_acc"], curves["val_acc"],
              "Accuracy", out_dir / f"{key}_acc.pdf", ylim=(0, 1))


def main():
    parser = argparse.ArgumentParser(prog="python -m src.compare.export_extractor_curves")
    parser.add_argument("--out", type=Path, default=ROOT / "tinygmm-tex" / "figures" / "extractor",
                        help="Output directory for the training-curve PDFs.")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs",
                        help="Lightning logs root containing <name>/version_*/metrics.csv.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for key, log_name, display in EXTRACTORS:
        curves = _per_epoch(args.logs / log_name)
        print(f"{display}: {len(curves)} epochs from {args.logs / log_name}")
        _plot(curves, key, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
