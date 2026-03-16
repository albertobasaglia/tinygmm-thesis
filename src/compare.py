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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

from models import SpeechExtractorModule, SpeechAutoencoder
from data import get_spectrograms


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class Adapter(ABC):
    """One-class adapter: fit on target embeddings, score new ones."""

    @abstractmethod
    def fit(self, train_emb: np.ndarray, val_emb: np.ndarray):
        """Train on target-class embeddings. Sets self.threshold."""

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
                 lr=1e-3, epochs=2000, device="cpu"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.threshold = None

    def fit(self, train_emb: np.ndarray, val_emb: np.ndarray):
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
    def __init__(self, n_components=3, covariance_type="full"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.threshold = None

    def fit(self, train_emb: np.ndarray, val_emb: np.ndarray):
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=42,
        )
        self._gmm.fit(train_emb)

        val_scores = self.score(val_emb)
        self.threshold = float(np.percentile(val_scores, 95))

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

    # AUC: label 0 = target (normal), 1 = other (anomaly)
    labels = np.concatenate([np.zeros(n_target), np.ones(n_other)])
    scores = np.concatenate([scores_target, scores_other])
    auc = roc_auc_score(labels, scores)

    return {
        "recall": hits / n_other,
        "false_alarm_rate": false_alarms / n_target,
        "accuracy": (hits + n_target - false_alarms) / (n_target + n_other),
        "auc": auc,
        "threshold": adapter.threshold,
    }


def print_results(results: dict[str, dict]):
    """Pretty-print comparison table."""
    max_name = max(len(n) for n in results) + 2
    header = f"{'Adapter':<{max_name}} {'Recall':>8} {'FAR':>8} {'Acc':>8} {'AUC':>8} {'Thresh':>10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<{max_name}} {m['recall']:>8.2%} {m['false_alarm_rate']:>8.2%} "
              f"{m['accuracy']:>8.2%} {m['auc']:>8.4f} {m['threshold']:>10.6f}")
    print("-" * len(header))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(results: dict, x: str, y: str, group_by: str = None, filter: str = None):
    """Plot sweep results with x and y from params/metrics, grouped into lines.

    Args:
        results  : output of the fit loop (includes _class and _params)
        x        : param name for the x-axis  (e.g. "n_components")
        y        : metric name for the y-axis  (e.g. "auc", "recall")
        group_by : param name for separate lines (e.g. "covariance_type")
        filter   : if set, only plot entries where _class == filter
    """
    rows = [m for m in results.values()
            if filter is None or m["_class"] == filter]

    # Group rows by the group_by param value (or single group if None)
    groups: dict[str, list] = {}
    for m in rows:
        key = str(m["_params"].get(group_by, "all")) if group_by else "all"
        groups.setdefault(key, []).append(m)

    fig, ax = plt.subplots()
    for label, entries in groups.items():
        entries.sort(key=lambda m: m["_params"].get(x, 0))
        xs = [e["_params"][x] for e in entries]
        ys = [e[y] for e in entries]
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    title = f"{filter or 'all'}: {y} vs {x}"
    if group_by:
        title += f" (grouped by {group_by})"
    ax.set_title(title)
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
    TRAIN_N = 48
    VAL_N = 16
    TEST_N = 500

    # --- Extract embeddings (done once, shared by all adapters) ---
    extractor = SpeechExtractorModule.load_from_checkpoint("best.ckpt")
    extractor.to(DEVICE).eval()

    print("Extracting training embeddings...")
    specs = get_spectrograms("./data", target_class="yes", n=TRAIN_N + VAL_N).to(DEVICE)
    with torch.no_grad():
        emb = extractor(specs, return_embedding=True).cpu().numpy()
    train_emb, val_emb = emb[:TRAIN_N], emb[TRAIN_N:]

    print("Extracting test embeddings...")
    specs_yes = get_spectrograms("./data", target_class="yes", n=TEST_N, subset="testing").to(DEVICE)
    specs_no = get_spectrograms("./data", target_class="no", n=TEST_N, subset="testing").to(DEVICE)
    with torch.no_grad():
        test_target = extractor(specs_yes, return_embedding=True).cpu().numpy()
        test_other = extractor(specs_no, return_embedding=True).cpu().numpy()

    # =================================================================
    # SWEEP CONFIG — edit these to control what gets compared
    #
    # Two ways to add adapters:
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
    configs = [
        ("Autoencoder", AutoencoderAdapter, {"device": DEVICE}),

        *sweep(GMMAdapter, {
            "n_components": [1, 2],
            "covariance_type": ["full", "diag", "spherical"],
        }),
    ]

    # --- Fit and evaluate ---
    results = {}
    for name, cls, kwargs in configs:
        print(f"  {name}...")
        adapter = cls(**kwargs)
        adapter.fit(train_emb, val_emb)
        results[name] = {
            **evaluate(adapter, test_target, test_other),
            "_class": cls.__name__,
            "_params": kwargs,
        }

    print()
    print_results(results)

    # =================================================================
    # PLOTS — edit x, y, and group_by to explore different views
    #
    # x        : param name to put on the x-axis
    # y        : metric name for the y-axis
    #            (recall, false_alarm_rate, accuracy, auc, threshold)
    # group_by : param name whose values become separate lines
    #            (if None, each config gets its own line)
    # filter   : only include entries where _class == this name
    # =================================================================
    # plot_sweep(results, filter="GMMAdapter", x="n_components", y="auc",       group_by="covariance_type")
    # plot_sweep(results, filter="GMMAdapter", x="n_components", y="recall",     group_by="covariance_type")
    # plot_sweep(results, filter="GMMAdapter", x="n_components", y="false_alarm_rate", group_by="covariance_type")
    # plt.show()


if __name__ == "__main__":
    main()
