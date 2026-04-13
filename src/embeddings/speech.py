import logging
from pathlib import Path

import numpy as np
import torch

from lib.models import SpeechExtractorModule
from lib.data import get_spectrograms

from .base import EmbeddingProvider

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "embeddings"


class SpeechEmbeddingProvider(EmbeddingProvider):
    """Embeddings from a pre-trained SpeechFeatureExtractor checkpoint."""

    def __init__(
        self,
        ckpt_path: str | Path,
        embedding_dim: int,
        data_dir: str | Path,
        target_class: str = "yes",
        other_classes: list[str] = None,
        device: str = "cpu",
    ):
        self.ckpt_path = Path(ckpt_path)
        self._embedding_dim = embedding_dim
        self.data_dir = str(data_dir)
        self.target_class = target_class
        self.other_classes = other_classes if other_classes is not None else ["no"]
        self.device = device

    @property
    def name(self) -> str:
        return f"speech_{self._embedding_dim}d_{self.target_class}"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _cache_path(self, train_n: int, test_n: int) -> Path:
        others = "_".join(sorted(self.other_classes))
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

        specs_train = get_spectrograms(
            self.data_dir, self.target_class, n=train_n, subset="training"
        )
        specs_target = get_spectrograms(
            self.data_dir, self.target_class, n=test_n, subset="testing"
        )
        specs_other = torch.cat([
            get_spectrograms(self.data_dir, cls, n=test_n, subset="testing")
            for cls in self.other_classes
        ])

        meta = torch.load(self.ckpt_path, weights_only=True)
        held_out = set(meta["hyper_parameters"].get("held_out_words") or [])
        for cls in [self.target_class, *self.other_classes]:
            if cls not in held_out:
                raise ValueError(
                    f"Class '{cls}' was NOT excluded from feature extractor training "
                    f"(held_out_words={held_out}). Sweep results would be invalid."
                )

        extractor = SpeechExtractorModule.load_from_checkpoint(self.ckpt_path)
        extractor.to(self.device).eval()

        with torch.no_grad():
            train_emb = extractor(specs_train.to(self.device), return_embedding=True).cpu().numpy()
            test_target = extractor(specs_target.to(self.device), return_embedding=True).cpu().numpy()
            test_other = extractor(specs_other.to(self.device), return_embedding=True).cpu().numpy()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, train_emb=train_emb, test_target=test_target, test_other=test_other)
        log.info("Embeddings cached to: %s", cache_path.name)

        return train_emb, test_target, test_other
