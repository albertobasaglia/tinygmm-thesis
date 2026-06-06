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
    The whole budget is used for fitting; the framework reports only
    threshold-free metrics (AUC / AUPRC / EER / acc_at_far5), so no
    threshold is computed or stored.
    """

    def __init__(self, train_n: int = None):
        self.train_n = train_n

    @property
    def threshold(self):
        raise AttributeError(
            "Adapter.threshold has been removed: the framework reports only "
            "threshold-free metrics. Score the data and aggregate over the "
            "score distribution instead."
        )

    @abstractmethod
    def fit(self, emb: np.ndarray):
        """Fit on at most self.train_n samples."""

    def _get_budget(self, emb: np.ndarray) -> np.ndarray:
        """Return the on-device budget: first train_n samples."""
        if self.train_n is not None:
            budget = emb[:self.train_n]
        else:
            budget = emb
        self._n_train = len(budget)
        return budget

    def _fitted_train_n(self) -> int:
        """Number of samples used by fit(), for post-fit cost accounting."""
        return max(1, getattr(self, "_n_train", self.train_n or 0))

    @abstractmethod
    def score(self, emb: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly score (higher = more anomalous)."""

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

    @abstractmethod
    def parameters(self) -> int:
        """Number of float values that must be stored on-device at inference.

        Counts the persistent state required to score a new sample
        (means, covariances, weights, prototype, stored neighbours, ...).
        """


class AutoencoderAdapter(Adapter):
    N_LOSS_CHECKPOINTS = 5

    def __init__(self, input_dim=32, hidden_dim=16, latent_dim=8,
                 lr=1e-3, epochs=2000, batch_size=8,
                 device="cpu", train_n=None, seed=None):
        super().__init__(train_n)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.train_loss_checkpoints: list[float] = []

    def fit(self, emb: np.ndarray):
        _t0 = time.perf_counter()
        train_emb = self._get_budget(emb)

        if self.seed is not None:
            torch.manual_seed(self.seed)
        model = SpeechAutoencoder(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_t = torch.tensor(train_emb, dtype=torch.float32, device=self.device)
        loader_gen = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        loader = DataLoader(TensorDataset(train_t), batch_size=self.batch_size,
                            shuffle=True, generator=loader_gen)

        ckpt_epochs = {
            round(i * self.epochs / self.N_LOSS_CHECKPOINTS)
            for i in range(1, self.N_LOSS_CHECKPOINTS + 1)
        }

        _t1 = time.perf_counter()
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

        model.eval()
        self._model = model
        _t2 = time.perf_counter()
        log.debug("AutoencoderAdapter fit: setup=%.3fs  train=%.3fs  total=%.3fs",
                  _t1 - _t0, _t2 - _t1, _t2 - _t0)

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
        n_train = self._fitted_train_n()
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
        # MSE: D subtracts + D squares + (D-1) adds ~ 3*D
        mse = 3 * D
        return linear + bias + mse

    def training_flops(self) -> int:
        D, H, L = self.input_dim, self.hidden_dim, self.latent_dim
        n_train = self._fitted_train_n()
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

    def parameters(self) -> int:
        D, H, L = self.input_dim, self.hidden_dim, self.latent_dim
        # Linear(D,H) + Linear(H,L) + Linear(L,H) + Linear(H,D), all with biases
        return (D+1)*H + (H+1)*L + (L+1)*H + (H+1)*D


class SmallAEAdapter(Adapter):
    """Small autoencoder: input_dim -> latent_dim (ReLU) -> input_dim."""

    N_LOSS_CHECKPOINTS = 5

    def __init__(self, input_dim=32, latent_dim=8,
                 lr=1e-3, epochs=2000, batch_size=8,
                 dropout_p=0.0, device="cpu", train_n=None, seed=None):
        super().__init__(train_n)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.device = device
        self.seed = seed
        self.train_loss_checkpoints: list[float] = []

    def fit(self, emb: np.ndarray):
        _t0 = time.perf_counter()
        train_emb = self._get_budget(emb)

        if self.seed is not None:
            torch.manual_seed(self.seed)
        model = SmallAutoencoder(self.input_dim, self.latent_dim, self.dropout_p).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_t = torch.tensor(train_emb, dtype=torch.float32, device=self.device)
        loader_gen = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        loader = DataLoader(TensorDataset(train_t), batch_size=self.batch_size,
                            shuffle=True, generator=loader_gen)

        ckpt_epochs = {
            round(i * self.epochs / self.N_LOSS_CHECKPOINTS)
            for i in range(1, self.N_LOSS_CHECKPOINTS + 1)
        }

        _t1 = time.perf_counter()
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

        model.eval()
        self._model = model
        _t2 = time.perf_counter()
        log.debug("SmallAEAdapter fit: setup=%.3fs  train=%.3fs  total=%.3fs",
                  _t1 - _t0, _t2 - _t1, _t2 - _t0)

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
        n_train = self._fitted_train_n()
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
        # MSE: D subtracts + D squares + (D-1) adds ~ 3*D
        mse = 3 * D
        return linear + bias + mse

    def training_flops(self) -> int:
        D, L = self.input_dim, self.latent_dim
        n_train = self._fitted_train_n()
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

    def parameters(self) -> int:
        D, L = self.input_dim, self.latent_dim
        # Linear(D,L) + Linear(L,D), both with biases
        return (D+1)*L + (L+1)*D


class GMMAdapter(Adapter):
    def __init__(self, n_components=3, covariance_type="full", train_n=None, seed=None):
        super().__init__(train_n)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.seed = seed

    def fit(self, emb: np.ndarray):
        train_emb = self._get_budget(emb)

        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=1e-4,
            random_state=self.seed,
        )
        self._gmm.fit(train_emb)

        self.avg_log_likelihood = self._gmm.score(train_emb)

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
        n_train = self._fitted_train_n()
        I = self._gmm.n_iter_
        if self.covariance_type == "spherical":
            per_iter = n_train * K * (2 * D + 2)  # E-step (D+1) + M-step (D+1)
        elif self.covariance_type == "diag":
            per_iter = n_train * K * 4 * D  # E-step (2D) + M-step (2D)
        else:
            per_iter = n_train * K * (2 * D**2 + 2 * D)
        return I * per_iter

    def inference_flops(self) -> int:
        """FLOPs to compute -log p(x) under a K-component GMM.

        The score is the negative log-likelihood:
            -log p(x) = -log sum_k [ pi_k * N(x | mu_k, Sigma_k) ]

        Decomposes into: Mahalanobis kernel + per-component weighting + logsumexp.
        """
        D = self._gmm.means_.shape[1]
        K = self.n_components

        # 1. Mahalanobis kernel, per component: maha_k^2 = (x - mu_k)^T @ prec_k @ (x - mu_k)
        #    where prec_k = Sigma_k^{-1} is the precision (inverse covariance).
        if self.covariance_type == "spherical":
            # prec_k is a scalar (1/sigma_k^2): maha^2 = prec_k * ||x - mu_k||^2
            #   D     sub   (x - mu_k)
            #   D     mul   squaring
            #   D-1   add   sum of squares
            #   1     mul   prec_k * sum
            kernel = K * (3 * D)
        elif self.covariance_type == "diag":
            # prec_k is a D-vector: maha^2 = sum_j prec_kj * (x_j - mu_kj)^2
            #   D     sub   (x - mu_k)
            #   D     mul   squaring
            #   D     mul   elementwise prec_k *
            #   D-1   add   sum
            kernel = K * (4 * D)
        else:
            # prec_k is a DxD matrix.
            #   D     sub   (x - mu_k)
            #   D^2   mul   mat-vec prec_k @ (x - mu_k)
            #   D^2   add   mat-vec accumulation
            #   D     mul   dot product with (x - mu_k)
            #   D-1   add   dot product sum
            kernel = K * (2 * D**2 + 2 * D)

        # 2. Per-component weighting: log_prob_k = log(pi_k) - 0.5 * maha_k^2
        #   1   mul   0.5 * maha^2
        #   1   sub   log(pi_k) - ...
        per_comp = 2 * K

        # 3. logsumexp reduction: log p(x) = m + log(sum_k exp(log_prob_k - m))
        #    The final negation for -log p(x) is a sign flip, not counted as a FLOP.
        #   K     sub   log_prob_k - m
        #   K     exp   exp(...)
        #   K-1   add   sum
        #   1     log   log(sum)
        #   1     add   m + log(sum)
        logsumexp = 3 * K + 1

        return kernel + per_comp + logsumexp

    def training_flops(self) -> int:
        """Estimated FLOPs for the fitted EM iterations.

        This accounts for repeated GMM scoring, responsibility normalization,
        and M-step parameter updates. It intentionally does not try to model
        sklearn's k-means initialization or low-level linear algebra constants.
        """
        D = self._gmm.means_.shape[1]
        K = self.n_components
        N = self._fitted_train_n()
        I = self._gmm.n_iter_

        e_step = N * (self.inference_flops() + K)  # +K for log-responsibility subtracts
        weights = N * K + K                        # nk sums + weight normalization
        means = K * D * (2 * N + 1)                # weighted sums + divide by nk

        if self.covariance_type == "spherical":
            cov = K * D * (4 * N + 1) + K          # diagonal variance accumulation, average over D
        elif self.covariance_type == "diag":
            cov = K * D * (4 * N + 1)              # diagonal variance accumulation
        else:
            cov = K * (N * D + 3 * N * D**2 + D**2 + D**3)

        return I * (e_step + weights + means + cov)

    def parameters(self) -> int:
        D = self._gmm.means_.shape[1]
        K = self.n_components
        means = K * D
        if self.covariance_type == "spherical":
            cov = K
        elif self.covariance_type == "diag":
            cov = K * D
        else:  # full
            cov = K * D * (D + 1) // 2
        weights = K
        return means + cov + weights


class PrototypeAdapter(Adapter):
    """Single prototype: fit = mean of enrollment, score = Euclidean distance.

    The simplest possible non-trivial baseline: collapse the enrolled
    samples to a single point and score by Euclidean distance to it.
    """

    def __init__(self, train_n=None):
        super().__init__(train_n)

    def fit(self, emb: np.ndarray):
        train_emb = self._get_budget(emb)
        self._prototype = train_emb.mean(axis=0)
        self._dim = train_emb.shape[1]

    def score(self, emb: np.ndarray) -> np.ndarray:
        return np.linalg.norm(emb - self._prototype, axis=1)

    def inference_macs(self) -> int:
        # D subtracts + D MACs for squared accumulation
        return 2 * self._dim

    def training_macs(self) -> int:
        # Sum of train_n D-vectors
        return self._fitted_train_n() * self._dim

    def inference_flops(self) -> int:
        # D sub + D mul + (D-1) add + 1 sqrt
        return 3 * self._dim

    def training_flops(self) -> int:
        # Mean: (N-1)*D adds + D divisions = N*D
        return self._fitted_train_n() * self._dim

    def parameters(self) -> int:
        return self._dim


class CosineAdapter(Adapter):
    """Single prototype: fit = mean of enrollment, score = 1 - cos(z, mean).

    Mirrors the TinySV-style prototype matcher used as the comparison
    point in the related-work / neural-collapse argument.
    """

    def __init__(self, train_n=None):
        super().__init__(train_n)

    def fit(self, emb: np.ndarray):
        train_emb = self._get_budget(emb)
        self._prototype = train_emb.mean(axis=0)
        self._proto_norm = float(np.linalg.norm(self._prototype))
        self._dim = train_emb.shape[1]

    def score(self, emb: np.ndarray) -> np.ndarray:
        dots = emb @ self._prototype
        norms = np.linalg.norm(emb, axis=1)
        cos_sim = dots / (norms * self._proto_norm + 1e-12)
        return 1.0 - cos_sim

    def inference_macs(self) -> int:
        # dot(z, prototype): D MACs;  ||z||^2: D MACs;  ||p|| precomputed
        return 2 * self._dim

    def training_macs(self) -> int:
        # Sum of train_n D-vectors + ||prototype||^2
        return self._fitted_train_n() * self._dim + self._dim

    def inference_flops(self) -> int:
        # dot: 2D-1; norm-squared: 2D-1; sqrt: 1; norms*proto_norm: 1; div: 1; 1 - cos: 1
        return 4 * self._dim + 3

    def training_flops(self) -> int:
        # Mean: N*D; prototype norm: D mul + (D-1) add + 1 sqrt = 2D
        return self._fitted_train_n() * self._dim + 2 * self._dim

    def parameters(self) -> int:
        # prototype (D) + cached ||prototype|| (1)
        return self._dim + 1


class KNNAdapter(Adapter):
    def __init__(self, k=5, metric="euclidean", train_n=None):
        super().__init__(train_n)
        self.k = k
        self.metric = metric

    def fit(self, emb: np.ndarray):
        train_emb = self._get_budget(emb)

        if self.k > len(train_emb):
            raise SkipConfig(
                f"k={self.k} > {len(train_emb)} training samples"
            )
        self._nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self._nn.fit(train_emb)

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
        if self.metric != "euclidean":
            raise NotImplementedError(
                "KNNAdapter FLOP accounting is implemented only for euclidean distance"
            )
        D = self._nn._fit_X.shape[1]
        n_stored = self._nn._fit_X.shape[0]
        # Distance-work estimate: D sub + D mul + (D-1) add + 1 sqrt = 3D.
        # Exact sklearn cost can differ if algorithm="auto" selects a tree.
        return n_stored * 3 * D

    def training_flops(self) -> int:
        return 0  # KNN just stores the data

    def parameters(self) -> int:
        # All stored training points
        n_stored, D = self._nn._fit_X.shape
        return n_stored * D
