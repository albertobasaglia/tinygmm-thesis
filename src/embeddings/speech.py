from pathlib import Path

import numpy as np
import torch

from lib.models import SpeechExtractorModule
from lib.data import get_spectrograms

from .base import EmbeddingProvider


class SpeechEmbeddingProvider(EmbeddingProvider):
    """Embeddings from a pre-trained SpeechFeatureExtractor checkpoint."""

    def __init__(
        self,
        ckpt_path: str | Path,
        embedding_dim: int,
        data_dir: str | Path,
        target_class: str = "yes",
        other_class: str = "no",
        device: str = "cpu",
    ):
        self.ckpt_path = Path(ckpt_path)
        self._embedding_dim = embedding_dim
        self.data_dir = str(data_dir)
        self.target_class = target_class
        self.other_class = other_class
        self.device = device

    @property
    def name(self) -> str:
        return f"speech_{self._embedding_dim}d"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_embeddings(
        self, train_n: int, test_n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: cache spectrograms so multiple providers sharing the same
        #       dataset don't reload and preprocess the same audio files.
        specs_train = get_spectrograms(
            self.data_dir, self.target_class, n=train_n, subset="training"
        )
        specs_target = get_spectrograms(
            self.data_dir, self.target_class, n=test_n, subset="testing"
        )
        specs_other = get_spectrograms(
            self.data_dir, self.other_class, n=test_n, subset="testing"
        )

        meta = torch.load(self.ckpt_path, weights_only=True)
        held_out = set(meta["hyper_parameters"].get("held_out_words") or [])
        for cls in (self.target_class, self.other_class):
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

        return train_emb, test_target, test_other
