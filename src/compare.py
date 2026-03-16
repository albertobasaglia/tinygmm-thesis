"""
Comparison framework for one-class adapters (autoencoder, GMM, ...).

Usage:
    cd src
    python compare.py

Edit the SWEEP dict in main() to control which configs get compared.
"""

from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_score, recall_score, f1_score

from models import SpeechExtractorModule, SpeechAutoencoder
from data import get_spectrograms


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class Adapter(ABC):
    """One-class adapter: fit on target embeddings, score new ones.

    Each adapter receives ALL available embeddings and must use only the
    first self.train_n of them — simulating the on-device budget.
    The adapter is responsible for any internal train/val split from
    that budget (e.g. for threshold calibration).
    """

    def __init__(self, train_n: int = None):
        self.train_n = train_n
        self.threshold = None

    @abstractmethod
    def fit(self, emb: np.ndarray):
        """Fit on at most self.train_n samples. Sets self.threshold."""

    def _get_budget(self, emb: np.ndarray) -> np.ndarray:
        """Return the on-device budget: first train_n samples."""
        if self.train_n is not None:
            return emb[:self.train_n]
        return emb

    @abstractmethod
    def score(self, emb: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly score (higher = more anomalous)."""

    def predict(self, emb: np.ndarray) -> np.ndarray:
        return self.score(emb) > self.threshold


# ---------------------------------------------------------------------------
# Autoencoder adapter
# ---------------------------------------------------------------------------

class AutoencoderAdapter(Adapter):
    def __init__(self, input_dim=32, hidden_dim=16, latent_dim=8,
                 lr=1e-3, epochs=2000, val_frac=0.25, device="cpu", train_n=None):
        super().__init__(train_n)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.val_frac = val_frac
        self.device = device

    def fit(self, emb: np.ndarray):
        budget = self._get_budget(emb)
        split = max(1, int(len(budget) * (1 - self.val_frac)))
        train_emb, val_emb = budget[:split], budget[split:]

        model = SpeechAutoencoder(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_t = torch.tensor(train_emb, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(train_t), batch_size=8, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for (x,) in loader:
                optimizer.zero_grad()
                criterion(model(x), x).backward()
                optimizer.step()

        model.eval()
        self._model = model

        val_scores = self.score(val_emb)
        self.threshold = float(np.percentile(val_scores, 95))

    def score(self, emb: np.ndarray) -> np.ndarray:
        x = torch.tensor(emb, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon = self._model(x)
            return torch.mean((x - recon) ** 2, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# GMM adapter
# ---------------------------------------------------------------------------

class GMMAdapter(Adapter):
    def __init__(self, n_components=3, covariance_type="full", train_n=None):
        super().__init__(train_n)
        self.n_components = n_components
        self.covariance_type = covariance_type

    def fit(self, emb: np.ndarray):
        budget = self._get_budget(emb)

        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=42,
        )
        self._gmm.fit(budget)

        train_scores = self.score(budget)
        self.threshold = float(np.percentile(train_scores, 95))

    def score(self, emb: np.ndarray) -> np.ndarray:
        # Negative log-likelihood: higher = more anomalous
        return -self._gmm.score_samples(emb)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(adapter: Adapter, target_emb: np.ndarray, other_emb: np.ndarray) -> dict:
    """Evaluate a fitted adapter. Returns dict of metrics."""
    scores_target = adapter.score(target_emb)
    scores_other = adapter.score(other_emb)

    preds_target = scores_target > adapter.threshold
    preds_other = scores_other > adapter.threshold

    n_target = len(target_emb)
    n_other = len(other_emb)
    false_alarms = preds_target.sum()
    hits = preds_other.sum()

    # label 0 = target (normal), 1 = other (anomaly)
    labels = np.concatenate([np.zeros(n_target), np.ones(n_other)])
    preds = np.concatenate([preds_target, preds_other]).astype(int)
    scores = np.concatenate([scores_target, scores_other])

    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # EER: operating point where FAR == FRR (1 - recall)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    precision = precision_score(labels, preds, zero_division=0)
    recall = hits / n_other
    f1 = f1_score(labels, preds, zero_division=0)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "false_alarm_rate": false_alarms / n_target,
        "accuracy": (hits + n_target - false_alarms) / (n_target + n_other),
        "auc": auc,
        "auprc": auprc,
        "eer": eer,
        "threshold": adapter.threshold,
    }




# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_far_recall(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """Scatter plot of operating points in FAR vs Recall space.

    Each (label, filter_dict) pair becomes one scatter series, with one point
    per row that matches the filter.  Useful for exposing degenerate configs
    (FAR≈1, recall≈1) versus well-calibrated ones.

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        subset = df.copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        ax.scatter(subset["false_alarm_rate"], subset["recall"], label=label, s=60)

    ax.set_xlabel("False Alarm Rate (FAR)")
    ax.set_ylabel("Recall (TPR)")
    ax.set_title("Operating points: Recall vs FAR")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_eer(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """EER vs train_n for selected configs.

    EER is threshold-free and directly comparable across adapters — lower is
    better.  A single line per config keeps the chart readable.

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        subset = df.copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        subset = subset.sort_values("train_n")
        ax.plot(subset["train_n"], subset["eer"], marker="o", label=label)

    ax.set_xlabel("train_n")
    ax.set_ylabel("EER")
    ax.set_title("Equal Error Rate vs training budget")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_auc_auprc(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """AUC-ROC (solid) and AUPRC (dashed) vs train_n on one figure.

    Plotting both metrics together exposes cases where AUC looks good but
    AUPRC is poor (e.g. degenerate GMM-full configs with FAR=1).

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
    """
    fig, ax = plt.subplots()
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [p["color"] for p in prop_cycle]

    for i, (label, where) in enumerate(lines):
        subset = df.copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        subset = subset.sort_values("train_n")
        c = colors[i % len(colors)]
        ax.plot(subset["train_n"], subset["auc"],   color=c, linestyle="-",  marker="o", label=f"{label} AUC-ROC")
        ax.plot(subset["train_n"], subset["auprc"], color=c, linestyle="--", marker="s", label=f"{label} AUPRC")

    ax.set_xlabel("train_n")
    ax.set_ylabel("Score")
    ax.set_title("AUC-ROC (—) vs AUPRC (--) by training budget")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_precision_recall_bar(df: pd.DataFrame, train_n: int,
                               lines: list[tuple[str, dict]]):
    """Grouped bar chart of precision and recall at a fixed train_n.

    Shows threshold calibration quality: a well-calibrated adapter has both
    high recall and high precision.  Degenerate configs (recall≈1, precision≈0.5)
    are immediately visible.

    Args:
        df      : results DataFrame
        train_n : the training budget to slice on
        lines   : list of (label, filter_dict) pairs
    """
    labels, precisions, recalls = [], [], []
    for label, where in lines:
        subset = df[df["train_n"] == train_n].copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        if subset.empty:
            continue
        row = subset.iloc[0]
        labels.append(label)
        precisions.append(row["precision"])
        recalls.append(row["recall"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, precisions, width, label="Precision")
    ax.bar(x + width / 2, recalls,    width, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision vs Recall at train_n={train_n}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def plot_f1(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """F1 vs train_n for selected configs.

    F1 penalises degenerate configs (recall=1, precision≈0.5 → F1≈0.66)
    while rewarding well-calibrated ones, making it a clean single-line
    comparison when the threshold matters.

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        subset = df.copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        subset = subset.sort_values("train_n")
        ax.plot(subset["train_n"], subset["f1"], marker="o", label=label)

    ax.set_xlabel("train_n")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs training budget")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_sweep(df: pd.DataFrame, x: str, y: str, group_by: str = None,
               filter: str = None, where: dict = None):
    """Plot sweep results with x and y as column names, grouped into lines.

    Args:
        df       : results DataFrame from main()
        x        : column name for the x-axis  (e.g. "n_components")
        y        : column name for the y-axis   (e.g. "auc", "recall")
        group_by : column name for separate lines (e.g. "covariance_type")
        filter   : if set, only plot rows where adapter == this name
        where    : extra column filters, e.g. {"covariance_type": "diag"}
    """
    subset = df.copy()
    if filter:
        subset = subset[subset["adapter"] == filter]
    if where:
        for k, v in where.items():
            subset = subset[subset[k] == v]

    fig, ax = plt.subplots()
    if group_by:
        for label, group in subset.groupby(group_by):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], marker="o", label=label)
    else:
        subset = subset.sort_values(x)
        ax.plot(subset[x], subset[y], marker="o")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    title = f"{filter or 'all'}: {y} vs {x}"
    if group_by:
        title += f" (grouped by {group_by})"
    if where:
        title += f" [{', '.join(f'{k}={v}' for k, v in where.items())}]"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_lines(df: pd.DataFrame, x: str, y: str,
               lines: list[tuple[str, dict]]):
    """Plot specific configs as named lines on one figure.

    Args:
        df    : results DataFrame
        x     : column for x-axis (e.g. "train_n")
        y     : column for y-axis (e.g. "auc")
        lines : list of (label, filter_dict) pairs — each becomes one line
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        subset = df.copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        subset = subset.sort_values(x)
        ax.plot(subset[x], subset[y], marker="o", label=label)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sweep(adapter_class: type, param_grid: dict) -> list[tuple[str, dict]]:
    """Expand a param grid into (name, kwargs) pairs.

    Example:
        sweep(GMMAdapter, {"n_components": [1, 3, 5], "covariance_type": ["full", "diag"]})
        → [("GMM n=1 cov=full", {...}), ("GMM n=1 cov=diag", {...}), ...]
    """
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    results = []
    for vals in combos:
        kwargs = dict(zip(keys, vals))
        tag = " ".join(f"{k}={v}" for k, v in kwargs.items())
        name = f"{adapter_class.__name__} {tag}"
        results.append((name, adapter_class, kwargs))
    return results


def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    TEST_N = 500

    # =================================================================
    # SWEEP CONFIG
    #
    # train_sizes : list of training set sizes to sweep over
    #               (embeddings are extracted once for the max size)
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
    # =================================================================
    TRAIN_N = 64

    train_n = [10, 15, 20, 25, 30, 40, 50, 100]

    configs = [
        *sweep(AutoencoderAdapter, {
            "train_n": train_n,
            "epochs": [100],
            "device": [DEVICE],
        }),

        *sweep(GMMAdapter, {
            "train_n": train_n,
            "n_components": [1, 2],
            "covariance_type": ["full", "diag"],
        }),
    ]

    # --- Extract embeddings once (enough for the largest train_n + val) ---
    extractor = SpeechExtractorModule.load_from_checkpoint("best.ckpt")
    extractor.to(DEVICE).eval()

    print("Extracting training embeddings...")
    specs = get_spectrograms("./data", target_class="yes", n=TRAIN_N).to(DEVICE)
    with torch.no_grad():
        train_emb = extractor(specs, return_embedding=True).cpu().numpy()

    print("Extracting test embeddings...")
    specs_yes = get_spectrograms("./data", target_class="yes", n=TEST_N, subset="testing").to(DEVICE)
    specs_no = get_spectrograms("./data", target_class="no", n=TEST_N, subset="testing").to(DEVICE)
    with torch.no_grad():
        test_target = extractor(specs_yes, return_embedding=True).cpu().numpy()
        test_other = extractor(specs_no, return_embedding=True).cpu().numpy()

    # --- Fit and evaluate ---
    rows = []
    for name, cls, kwargs in configs:
        print(f"  {name}...")
        adapter = cls(**kwargs)
        adapter.fit(train_emb)
        rows.append({"adapter": cls.__name__, **kwargs, **evaluate(adapter, test_target, test_other)})

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    # =================================================================
    # PLOTS
    #
    # plot_sweep: one group_by column becomes separate lines
    # plot_lines: pick exact configs as named lines on one plot
    # plot_far_recall: operating point scatter (FAR vs Recall)
    # plot_eer: EER vs train_n (threshold-free)
    # plot_auc_auprc: AUC-ROC vs AUPRC comparison
    # plot_precision_recall_bar: precision/recall bar at fixed train_n
    # plot_f1: F1 vs train_n
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

    # # Operating point scatter — exposes degenerate full-cov configs
    # plot_far_recall(df, lines=lines)

    # # Threshold-free comparison: EER (lower = better)
    # plot_eer(df, lines=lines_good)

    # # AUC-ROC vs AUPRC — flags cases where AUC flatters a config
    # plot_auc_auprc(df, lines=lines_good)

    # # Precision & Recall at one representative budget
    # plot_precision_recall_bar(df, train_n=50, lines=lines)

    # # F1 penalises degenerate configs, clean single-number comparison
    # plot_f1(df, lines=lines_good)

    plt.show()

    return df


if __name__ == "__main__":
    main()
