from abc import ABC, abstractmethod
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from lib.models import SpeechAutoencoder, LinearAutoencoder


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

    @abstractmethod
    def inference_macs(self) -> int:
        """MACs to score a single sample."""

    @abstractmethod
    def training_macs(self) -> int:
        """Total MACs for the fit() call."""


class AutoencoderAdapter(Adapter):
    def __init__(self, input_dim=32, hidden_dim=16, latent_dim=8,
                 lr=1e-3, epochs=2000, batch_size=8, val_frac=0.25, device="cpu", train_n=None):
        super().__init__(train_n)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
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
        loader = DataLoader(TensorDataset(train_t), batch_size=self.batch_size, shuffle=True)

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

    def inference_macs(self) -> int:
        D, H, L = self.input_dim, self.hidden_dim, self.latent_dim
        linear = D*H + H*L + L*H + H*D
        mse = D  # subtract, square, accumulate
        return linear + mse

    def training_macs(self) -> int:
        D, H, L = self.input_dim, self.hidden_dim, self.latent_dim
        n_train = max(1, int(self.train_n * (1 - self.val_frac)))
        F = D*H + H*L + L*H + H*D          # forward linear MACs
        P = (D+1)*H + (H+1)*L + (L+1)*H + (H+1)*D  # parameters (w+b)
        per_epoch = n_train * (3*F + 2*D) + math.ceil(n_train / self.batch_size) * 5 * P
        return self.epochs * per_epoch


class LinearAEAdapter(Adapter):
    """Linear autoencoder: input_dim → latent_dim → input_dim (no activations)."""

    def __init__(self, input_dim=32, latent_dim=8,
                 lr=1e-3, epochs=2000, batch_size=8, val_frac=0.25, device="cpu", train_n=None):
        super().__init__(train_n)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.device = device

    def fit(self, emb: np.ndarray):
        budget = self._get_budget(emb)
        split = max(1, int(len(budget) * (1 - self.val_frac)))
        train_emb, val_emb = budget[:split], budget[split:]

        model = LinearAutoencoder(self.input_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_t = torch.tensor(train_emb, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(train_t), batch_size=self.batch_size, shuffle=True)

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

    def inference_macs(self) -> int:
        D, L = self.input_dim, self.latent_dim
        linear = D*L + L*D  # encoder + decoder
        mse = D
        return linear + mse

    def training_macs(self) -> int:
        D, L = self.input_dim, self.latent_dim
        n_train = max(1, int(self.train_n * (1 - self.val_frac)))
        F = D*L + L*D                      # forward linear MACs
        P = (D+1)*L + (L+1)*D              # parameters (w+b)
        per_epoch = n_train * (3*F + 2*D) + math.ceil(n_train / self.batch_size) * 5 * P
        return self.epochs * per_epoch


class GMMAdapter(Adapter):
    def __init__(self, n_components=3, covariance_type="full", val_frac=0.25,
                 threshold_percentile=95, train_n=None):
        super().__init__(train_n)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.val_frac = val_frac
        self.threshold_percentile = threshold_percentile

    def fit(self, emb: np.ndarray):
        budget = self._get_budget(emb)
        split = max(1, int(len(budget) * (1 - self.val_frac)))
        train_emb, val_emb = budget[:split], budget[split:]

        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=1e-4,
        )
        self._gmm.fit(train_emb)

        val_scores = self.score(val_emb)
        self.threshold = float(np.percentile(val_scores, self.threshold_percentile))

    def score(self, emb: np.ndarray) -> np.ndarray:
        # Negative log-likelihood: higher = more anomalous
        return -self._gmm.score_samples(emb)

    def inference_macs(self) -> int:
        D = self._gmm.means_.shape[1]
        K = self.n_components
        if self.covariance_type == "diag":
            return K * 2 * D  # element-wise precision multiply + squared norm
        return K * (D**2 + D)  # dense mat-vec + squared norm

    def training_macs(self) -> int:
        D = self._gmm.means_.shape[1]
        K = self.n_components
        n_train = max(1, int(self.train_n * (1 - self.val_frac)))
        I = self._gmm.n_iter_
        if self.covariance_type == "diag":
            per_iter = n_train * K * 4 * D  # E-step (2D) + M-step (2D)
        else:
            per_iter = n_train * K * (2 * D**2 + 2 * D)
        return I * per_iter


class KNNAdapter(Adapter):
    def __init__(self, k=5, metric="euclidean", val_frac=0.25, train_n=None):
        super().__init__(train_n)
        self.k = k
        self.metric = metric
        self.val_frac = val_frac

    def fit(self, emb: np.ndarray):
        budget = self._get_budget(emb)
        split = max(1, int(len(budget) * (1 - self.val_frac)))
        train_emb, val_emb = budget[:split], budget[split:]

        self._nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self._nn.fit(train_emb)

        val_scores = self.score(val_emb)
        self.threshold = float(np.percentile(val_scores, 95))

    def score(self, emb: np.ndarray) -> np.ndarray:
        # Distance to k-th nearest neighbor (higher = more anomalous)
        distances, _ = self._nn.kneighbors(emb)
        return distances[:, -1]

    def inference_macs(self) -> int:
        D = self._nn._fit_X.shape[1]
        n_stored = self._nn._fit_X.shape[0]
        return n_stored * 2 * D  # euclidean distance to all stored points

    def training_macs(self) -> int:
        return 0  # KNN just stores the data
