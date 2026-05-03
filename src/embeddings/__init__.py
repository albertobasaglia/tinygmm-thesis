import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # ensure src/ is on the path

from .base import EmbeddingProvider
from .speech import SpeechEmbeddingProvider
from .tabular import TabularEmbeddingProvider
