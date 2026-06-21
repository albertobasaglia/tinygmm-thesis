from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Provides train/test embeddings for one-class adapter evaluation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'speech_16d'), used in results."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""

    @abstractmethod
    def get_embeddings(
        self, train_n: int, test_n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_target, test_target, test_other) numpy arrays.

        Args:
            train_n: Number of target-class training embeddings to return.
            test_n: Number of test embeddings per class.

        Returns:
            train_target: (train_n, embedding_dim) target-class embeddings for fitting.
            test_target:  (test_n, embedding_dim) held-out target-class embeddings.
            test_other:   (test_n, embedding_dim) non-target embeddings.
        """

    def standardize(
        self, fit_on: np.ndarray, *arrays: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Standardize arrays using statistics estimated from `fit_on` only.

        `fit_on` is the budgeted enrollment subset the adapter will actually
        be fitted on, so any scaler must be estimated from it (and applied to
        the evaluation arrays) to honor the few-shot enrollment budget — never
        from the full pool.

        The default is the identity transform: providers whose features are
        already on a common scale (e.g. instance-normalized neural embeddings)
        do not override it. Providers over raw heterogeneous features override
        this to fit a scaler on the enrollment subset.

        Returns (fit_on, *arrays), each transformed.
        """
        return (fit_on, *arrays)
