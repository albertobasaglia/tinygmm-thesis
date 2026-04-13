from abc import ABC, abstractmethod
import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from lib.models import SpeechAutoencoder, SmallAutoencoder

log = logging.getLogger(__name__)


class SkipConfig(Exception):
    """Raised when a config is invalid for the given parameters."""
    pass


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

    @abstractmethod
    def inference_flops(self) -> int:
        """FLOPs to score a single sample."""

    @abstractmethod
    def training_flops(self) -> int:
        """Total FLOPs for the fit() call."""


class AutoencoderAdapter(Adapter):
    N_LOSS_CHECKPOINTS = 5

    def __init__(self, input_dim=32, hidden_dim=16, latent_dim=8,
                 lr=1e-3, epochs=2000, batch_size=8, val_frac=0.25,
                 threshold_mode="val", device="cpu", train_n=None):
        super().__init__(train_n)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.threshold_mode = threshold_mode
        self.device = device
        self.val_loss_checkpoints: list[float] = []
        self.train_loss_checkpoints: list[float] = []

    def fit(self, emb: np.ndarray):
        _t0 = time.perf_counter()
        budget = self._get_budget(emb)
        if self.threshold_mode == "train":
            train_emb = budget
            val_emb = None
        else:
            split = max(1, int(len(budget) * (1 - self.val_frac)))
            train_emb, val_emb = budget[:split], budget[split:]

        model = SpeechAutoencoder(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_t = torch.tensor(train_emb, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(train_t), batch_size=self.batch_size, shuffle=True)

        if val_emb is not None:
            val_t = torch.tensor(val_emb, dtype=torch.float32, device=self.device)

        # Epochs at which to record the validation loss (1-indexed, evenly spaced)
        ckpt_epochs = {
            round(i * self.epochs / self.N_LOSS_CHECKPOINTS)
            for i in range(1, self.N_LOSS_CHECKPOINTS + 1)
        }

        _t1 = time.perf_counter()
        self.val_loss_checkpoints = []
        self.train_loss_checkpoints = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = train_t.new_zeros(())
            for (x,) in loader:
                optimizer.zero_grad()
                loss = criterion(model(x), x)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    epoch_loss += loss * len(x)
            if epoch in ckpt_epochs:
                self.train_loss_checkpoints.append((epoch_loss / len(train_t)).item())
                if val_emb is not None:
                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(val_t), val_t).item()
                    self.val_loss_checkpoints.append(val_loss)

        model.eval()
        self._model = model
        _t2 = time.perf_counter()

        threshold_scores = self.score(train_emb if val_emb is None else val_emb)
        self.threshold = float(np.percentile(threshold_scores, 95))
        _t3 = time.perf_counter()
        log.debug("AutoencoderAdapter fit: setup=%.3fs  train=%.3fs  threshold=%.3fs  total=%.3fs",
                  _t1 - _t0, _t2 - _t1, _t3 - _t2, _t3 - _t0)

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
        frac = 0.0 if self.threshold_mode == "train" else self.val_frac
        n_train = max(1, int(self.train_n * (1 - frac)))
        F = D*H + H*L + L*H + H*D          # forward linear MACs
        P = (D+1)*H + (H+1)*L + (L+1)*H + (H+1)*D  # parameters (w+b)
        per_epoch = n_train * (3*F + 2*D) + math.ceil(n_train / self.batch_size) * 5 * P
        return self.epochs * per_epoch

    def inference_flops(self) -> int:
        D, H, L = self.input_dim, self.hidden_dim, self.latent_dim
        # Linear layers: 2 FLOPs per MAC (multiply + accumulate)
        linear = 2 * (D*H + H*L + L*H + H*D)
        # Bias additions: one add per output unit per layer
        bias = H + L + H + D
        # MSE: D subtracts + D squares + (D-1) adds ≈ 3*D
        mse = 3 * D
        return linear + bias + mse

    def training_flops(self) -> int:
        D, H, L = self.input_dim, self.hidden_dim, self.latent_dim
        frac = 0.0 if self.threshold_mode == "train" else self.val_frac
        n_train = max(1, int(self.train_n * (1 - frac)))
        F = D*H + H*L + L*H + H*D              # forward linear MACs
        B = H + L + H + D                       # bias adds (forward)
        P = (D+1)*H + (H+1)*L + (L+1)*H + (H+1)*D  # parameters (w+b)
        # Forward:  2*F (linear FLOPs) + B (bias) + 3*D (MSE: sub+sq+accum)
        # Backward: 4*F (linear, ~2x forward) + D (MSE gradient)
        # Adam per parameter per step: 16 FLOPs
        #   m = b1*m + (1-b1)*g            -> 3 (2 mul + 1 add)
        #   v = b2*v + (1-b2)*g^2          -> 4 (1 sq + 2 mul + 1 add)
        #   m_hat = m/(1-b1^t)             -> 1 (div)
        #   v_hat = v/(1-b2^t)             -> 1 (div)
        #   w -= lr*m_hat/(sqrt(v_hat)+e)  -> 5 (sqrt + add + div + mul + sub)
        #   weight decay: g += wd*w        -> 2 (mul + add)
        per_sample = 6*F + B + 4*D
        adam_per_step = 16 * P
        per_epoch = n_train * per_sample + math.ceil(n_train / self.batch_size) * adam_per_step
        return self.epochs * per_epoch


class SmallAEAdapter(Adapter):
    """Small autoencoder: input_dim → latent_dim (ReLU) → input_dim."""

    N_LOSS_CHECKPOINTS = 5

    def __init__(self, input_dim=32, latent_dim=8,
                 lr=1e-3, epochs=2000, batch_size=8, val_frac=0.25,
                 threshold_mode="val", dropout_p=0.0, device="cpu", train_n=None):
        super().__init__(train_n)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.threshold_mode = threshold_mode
        self.dropout_p = dropout_p
        self.device = device
        self.val_loss_checkpoints: list[float] = []
        self.train_loss_checkpoints: list[float] = []

    def fit(self, emb: np.ndarray):
        _t0 = time.perf_counter()
        budget = self._get_budget(emb)
        if self.threshold_mode == "train":
            train_emb = budget
            val_emb = None
        else:
            split = max(1, int(len(budget) * (1 - self.val_frac)))
            train_emb, val_emb = budget[:split], budget[split:]

        model = SmallAutoencoder(self.input_dim, self.latent_dim, self.dropout_p).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_t = torch.tensor(train_emb, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(train_t), batch_size=self.batch_size, shuffle=True)

        if val_emb is not None:
            val_t = torch.tensor(val_emb, dtype=torch.float32, device=self.device)

        ckpt_epochs = {
            round(i * self.epochs / self.N_LOSS_CHECKPOINTS)
            for i in range(1, self.N_LOSS_CHECKPOINTS + 1)
        }

        _t1 = time.perf_counter()
        self.val_loss_checkpoints = []
        self.train_loss_checkpoints = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = train_t.new_zeros(())
            for (x,) in loader:
                optimizer.zero_grad()
                loss = criterion(model(x), x)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    epoch_loss += loss * len(x)
            if epoch in ckpt_epochs:
                self.train_loss_checkpoints.append((epoch_loss / len(train_t)).item())
                if val_emb is not None:
                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(val_t), val_t).item()
                    self.val_loss_checkpoints.append(val_loss)

        model.eval()
        self._model = model
        _t2 = time.perf_counter()

        threshold_scores = self.score(train_emb if val_emb is None else val_emb)
        self.threshold = float(np.percentile(threshold_scores, 95))
        _t3 = time.perf_counter()
        log.debug("SmallAEAdapter fit: setup=%.3fs  train=%.3fs  threshold=%.3fs  total=%.3fs",
                  _t1 - _t0, _t2 - _t1, _t3 - _t2, _t3 - _t0)

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
        frac = 0.0 if self.threshold_mode == "train" else self.val_frac
        n_train = max(1, int(self.train_n * (1 - frac)))
        F = D*L + L*D                      # forward linear MACs
        P = (D+1)*L + (L+1)*D              # parameters (w+b)
        per_epoch = n_train * (3*F + 2*D) + math.ceil(n_train / self.batch_size) * 5 * P
        return self.epochs * per_epoch

    def inference_flops(self) -> int:
        D, L = self.input_dim, self.latent_dim
        # Linear layers: 2 FLOPs per MAC
        linear = 2 * (D*L + L*D)
        # Bias additions: one add per output unit per layer
        bias = L + D
        # MSE: D subtracts + D squares + (D-1) adds ≈ 3*D
        mse = 3 * D
        return linear + bias + mse

    def training_flops(self) -> int:
        D, L = self.input_dim, self.latent_dim
        frac = 0.0 if self.threshold_mode == "train" else self.val_frac
        n_train = max(1, int(self.train_n * (1 - frac)))
        F = D*L + L*D                          # forward linear MACs
        B = L + D                               # bias adds (forward)
        P = (D+1)*L + (L+1)*D                  # parameters (w+b)
        # Forward:  2*F (linear FLOPs) + B (bias) + 3*D (MSE)
        # Backward: 4*F (linear, ~2x forward) + D (MSE gradient)
        # Adam: 16 FLOPs per parameter per step (see AutoencoderAdapter)
        per_sample = 6*F + B + 4*D
        adam_per_step = 16 * P
        per_epoch = n_train * per_sample + math.ceil(n_train / self.batch_size) * adam_per_step
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

        self.avg_log_likelihood = self._gmm.score(train_emb)

        val_scores = self.score(val_emb)
        self.threshold = float(np.percentile(val_scores, self.threshold_percentile))

    def score(self, emb: np.ndarray) -> np.ndarray:
        # Negative log-likelihood: higher = more anomalous
        return -self._gmm.score_samples(emb)

    def inference_macs(self) -> int:
        D = self._gmm.means_.shape[1]
        K = self.n_components
        if self.covariance_type == "spherical":
            return K * D  # dot product for squared norm
        if self.covariance_type == "diag":
            return K * D  # weighted accumulation: prec_i * delta_i^2 summed
        return K * D**2  # dense precision mat-vec

    def training_macs(self) -> int:
        D = self._gmm.means_.shape[1]
        K = self.n_components
        n_train = max(1, int(self.train_n * (1 - self.val_frac)))
        I = self._gmm.n_iter_
        if self.covariance_type == "spherical":
            per_iter = n_train * K * (2 * D + 2)  # E-step (D+1) + M-step (D+1)
        elif self.covariance_type == "diag":
            per_iter = n_train * K * 4 * D  # E-step (2D) + M-step (2D)
        else:
            per_iter = n_train * K * (2 * D**2 + 2 * D)
        return I * per_iter

    def inference_flops(self) -> int:
        D = self._gmm.means_.shape[1]
        K = self.n_components
        if self.covariance_type == "spherical":
            # delta = x - mu (D sub), delta^2 (D mul), sum (D-1 add), /var (1 div)
            return K * (3 * D)
        if self.covariance_type == "diag":
            # Per component:
            #   delta = x - mu:           D subtractions
            #   delta^2:                  D multiplications
            #   prec * delta^2:           D multiplications
            #   sum over D:               (D-1) additions
            # Total: ~4D FLOPs per component
            return K * 4 * D
        # full: delta (D sub), prec @ delta (D^2 mul + D^2 add), dot (D mul + D-1 add)
        return K * (2 * D**2 + 3 * D)

    def training_flops(self) -> int:
        # EM algorithm is multiply-accumulate dominated: 2 FLOPs per MAC
        return 2 * self.training_macs()


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

        if self.k > len(train_emb):
            raise SkipConfig(
                f"k={self.k} > {len(train_emb)} training samples"
            )
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

    def inference_flops(self) -> int:
        # Euclidean distance: 2 FLOPs per MAC (multiply + accumulate)
        return 2 * self.inference_macs()

    def training_flops(self) -> int:
        return 0  # KNN just stores the data
