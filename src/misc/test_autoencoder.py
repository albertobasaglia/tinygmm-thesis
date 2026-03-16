from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch as T

from lib.models import SpeechExtractorModule, SpeechAnomalyModule
from lib.data import get_spectrograms

ROOT = Path(__file__).parent.parent.parent

# --- 1. Setup and Model Loading ---
DEVICE = "mps" if T.backends.mps.is_available() else "cpu"

extractor = SpeechExtractorModule.load_from_checkpoint(ROOT / "best.ckpt")
anomalymodule = SpeechAnomalyModule.load_from_checkpoint(ROOT / "best_anomaly_model.ckpt")

anomalymodule.to(DEVICE).eval()
extractor.to(DEVICE).eval()

threshold = anomalymodule.computed_threshold.item()

print("Loading 1,000 test samples...")
TEST_N = 500
specs_yes = get_spectrograms(str(ROOT / "data"), target_class="yes", n=TEST_N, subset="testing").to(DEVICE)
specs_no  = get_spectrograms(str(ROOT / "data"), target_class="no",  n=TEST_N, subset="testing").to(DEVICE)

with T.no_grad():
    emb_yes = extractor(specs_yes, return_embedding=True)
    scores_yes = anomalymodule.get_anomaly_score(emb_yes).cpu().numpy()
    preds_yes = scores_yes > threshold # True = Flagged as Anomaly

    emb_no = extractor(specs_no, return_embedding=True)
    scores_no = anomalymodule.get_anomaly_score(emb_no).cpu().numpy()
    preds_no = scores_no > threshold   # True = Flagged as Anomaly

hits = np.sum(preds_no)          # Correct: 'no' detected as anomaly
misses = TEST_N - hits           # Incorrect: 'no' let through as 'yes'
false_alarms = np.sum(preds_yes) # Incorrect: 'yes' flagged as anomaly
correct_normals = TEST_N - false_alarms

print("-" * 30)
print(f"TEST RESULTS (Threshold={threshold:.4f})")
print("-" * 30)
print(f"Anomaly Detection (Recall): {hits}/{TEST_N} ({hits/TEST_N:.2%})")
print(f"False Alarm Rate:           {false_alarms}/{TEST_N} ({false_alarms/TEST_N:.2%})")
print(f"Overall Accuracy:           {(hits + correct_normals) / (TEST_N * 2):.2%}")
print("-" * 30)

# --- Visualization ---
plt.figure(figsize=(12, 6))

plt.hist(scores_yes, bins=50, alpha=0.6, label='Normal (Yes)', color='#3498db', density=True)
plt.hist(scores_no, bins=50, alpha=0.6, label='Anomaly (No)', color='#e74c3c', density=True)

plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold (95th Perc)')

plt.title("One-Class Classification: Reconstruction Error Distribution", fontsize=14)
plt.xlabel("Reconstruction MSE (Anomaly Score)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
