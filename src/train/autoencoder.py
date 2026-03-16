from pathlib import Path
import torch as T
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import numpy as np

from lib.models import SpeechExtractorModule, SpeechAnomalyModule
from lib.data import get_spectrograms

ROOT = Path(__file__).parent.parent.parent

extractor = SpeechExtractorModule.load_from_checkpoint(ROOT / "best.ckpt")
extractor.eval()


TRAIN_N = 48
VAL_N = 16

specs = get_spectrograms(str(ROOT / "data"), target_class="yes", n=TRAIN_N+VAL_N).to("mps")

with T.no_grad():
    out = extractor(specs, return_embedding=True)

out_train, out_val = out[:TRAIN_N], out[TRAIN_N:]

dataset_train = TensorDataset(out_train)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

dataset_val = TensorDataset(out_val)
dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=False)

anomalymodule = SpeechAnomalyModule()

logger = CSVLogger("logs", name="anomaly_model")

MAX_EPOCHS = 2000

trainer = L.Trainer(max_epochs=MAX_EPOCHS, accelerator="mps", logger=logger)
trainer.fit(anomalymodule, dataloader_train, dataloader_val)

# Extracting the threshold

anomalymodule.eval()
with T.no_grad():
    out_val = out_val.to("cpu")
    val_scores = anomalymodule.get_anomaly_score(out_val).cpu().numpy()

    computed_threshold = np.percentile(val_scores, 95)

    anomalymodule.computed_threshold = T.tensor(computed_threshold)

trainer.save_checkpoint(str(ROOT / "best_anomaly_model.ckpt"))

print(f"\nFinal Training Threshold: {computed_threshold:.6f}")
