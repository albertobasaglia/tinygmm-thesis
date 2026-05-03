import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import EmbeddingProvider

log = logging.getLogger(__name__)


class TabularEmbeddingProvider(EmbeddingProvider):
    """Returns raw tabular features as embeddings, skipping any feature extractor.

    Loads a CSV or parquet file once, splits rows by a label column into
    target and adversarial pools, and returns numpy arrays compatible with
    the adapter sweep.
    """

    def __init__(
        self,
        data_path: str | Path,
        label_column: str,
        target_class,
        other_classes: list | None = None,
        feature_columns: list[str] | None = None,
        test_split_seed: int = 0,
        scale: bool = True,
    ):
        self.data_path = Path(data_path)
        self.label_column = label_column
        self.target_class = target_class
        self.test_split_seed = test_split_seed
        self.scale = scale

        df = self._load(self.data_path)
        if label_column not in df.columns:
            raise ValueError(
                f"label_column '{label_column}' not in {self.data_path.name}; "
                f"have {list(df.columns)}"
            )

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != label_column]
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"feature_columns missing from data: {missing}")
        self.feature_columns = feature_columns

        labels = df[label_column]
        if other_classes is None:
            other_classes = sorted(v for v in labels.unique() if v != target_class)
        self.other_classes = list(other_classes)

        self._target_df = df.loc[labels == target_class, feature_columns].reset_index(drop=True)
        self._other_df = df.loc[labels.isin(self.other_classes), feature_columns].reset_index(drop=True)

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file type: {path.suffix} (use .parquet or .csv)")

    @property
    def name(self) -> str:
        return f"tabular_{self.data_path.stem}_{self.target_class}"

    @property
    def embedding_dim(self) -> int:
        return len(self.feature_columns)

    def get_embeddings(
        self, train_n: int, test_n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_target = len(self._target_df)
        n_other = len(self._other_df)
        if n_target < train_n + test_n:
            raise ValueError(
                f"target class '{self.target_class}' has {n_target} rows, "
                f"need {train_n} train + {test_n} test"
            )
        if n_other < test_n:
            raise ValueError(
                f"adversarial pool has {n_other} rows, need {test_n} test"
            )

        rng = np.random.default_rng(self.test_split_seed)

        target_perm = rng.permutation(n_target)
        test_idx = target_perm[:test_n]
        train_pool_idx = target_perm[test_n:]
        train_idx = train_pool_idx[:train_n]

        other_idx = rng.choice(n_other, size=test_n, replace=False)

        train_target = self._target_df.iloc[train_idx].to_numpy(dtype=np.float32)
        test_target = self._target_df.iloc[test_idx].to_numpy(dtype=np.float32)
        test_other = self._other_df.iloc[other_idx].to_numpy(dtype=np.float32)

        if self.scale:
            scaler = StandardScaler().fit(train_target)
            train_target = scaler.transform(train_target).astype(np.float32)
            test_target = scaler.transform(test_target).astype(np.float32)
            test_other = scaler.transform(test_other).astype(np.float32)

        return train_target, test_target, test_other
