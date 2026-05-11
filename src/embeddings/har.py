import logging
from pathlib import Path

import numpy as np
import torch

from lib.models import HARExtractorModule
from lib.data import get_windows

from .base import EmbeddingProvider

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "embeddings"


class HAREmbeddingProvider(EmbeddingProvider):
    """Embeddings from a pre-trained HARFeatureExtractor checkpoint (WISDM-2019)."""

    def __init__(
        self,
        ckpt_path: str | Path,
        embedding_dim: int,
        data_dir: str | Path,
        target_class: int,
        other_classes: list[int] = None,
        device: str = "cpu",
    ):
        self.ckpt_path = Path(ckpt_path)
        self._embedding_dim = embedding_dim
        self.data_dir = str(data_dir)
        self.target_class = int(target_class)
        self.other_classes = [int(c) for c in (other_classes if other_classes is not None else [])]
        self.device = device

    @property
    def name(self) -> str:
        return f"har_{self._embedding_dim}d_{self.target_class}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _cache_path(self, train_n: int, test_n: int) -> Path:
        others = "_".join(str(c) for c in sorted(self.other_classes))
        key = f"{self.ckpt_path.stem}__{self.target_class}__{others}__train{train_n}_test{test_n}.npz"
        return _CACHE_DIR / key

    def get_embeddings(
        self, train_n: int, test_n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_path = self._cache_path(train_n, test_n)
        if cache_path.exists():
            log.info("Loading embeddings from cache: %s", cache_path.name)
            data = np.load(cache_path)
            return data["train_emb"], data["test_target"], data["test_other"]

        windows_train = get_windows(
            self.data_dir, self.target_class, n=train_n, subset="training"
        )
        windows_target = get_windows(
            self.data_dir, self.target_class, n=test_n, subset="testing"
        )
        windows_other = torch.cat([
            get_windows(self.data_dir, cls, n=test_n, subset="testing")
            for cls in self.other_classes
        ])

        meta = torch.load(self.ckpt_path, weights_only=True, map_location="cpu")
        held_out = set(int(s) for s in (meta["hyper_parameters"].get("held_out_subjects") or []))
        for cls in [self.target_class, *self.other_classes]:
            if cls not in held_out:
                raise ValueError(
                    f"Subject '{cls}' was NOT excluded from feature extractor training "
                    f"(held_out_subjects={sorted(held_out)}). Sweep results would be invalid."
                )

        extractor = HARExtractorModule.load_from_checkpoint(self.ckpt_path)
        extractor.to(self.device).eval()

        with torch.no_grad():
            train_emb = extractor(windows_train.to(self.device), return_embedding=True).cpu().numpy()
            test_target = extractor(windows_target.to(self.device), return_embedding=True).cpu().numpy()
            test_other = extractor(windows_other.to(self.device), return_embedding=True).cpu().numpy()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, train_emb=train_emb, test_target=test_target, test_other=test_other)
        log.info("Embeddings cached to: %s", cache_path.name)

        return train_emb, test_target, test_other
