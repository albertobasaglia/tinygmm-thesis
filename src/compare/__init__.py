import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # ensure src/ is on the path

from .adapters import Adapter, AutoencoderAdapter, GMMAdapter
from .metrics import evaluate
from .sweep import sweep
from .plots import (
    plot_lines,
    plot_sweep,
    plot_far_recall,
    plot_eer,
    plot_precision_recall_bar,
    plot_f1,
    plot_eer_by_dim,
    plot_eer_train_n_by_dim,
)
