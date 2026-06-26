# tinygmm

Adaptive Probabilistic Models for TinyML-based Wearable Personalization (DTU M.Sc. thesis, Alberto Basaglia).

Investigates whether **Gaussian Mixture Models** can act as a TinyML-friendly adaptive layer for on-device personalization. The GMM is compared against several baselines — k-NN, prototype, cosine, and a small autoencoder — on few-shot one-class / anomaly-detection tasks.

The system is a **pre-trained DNN feature extractor** + an **adaptive layer** (GMM or a baseline) where the few-shot personalization happens.

## Setup

```bash
uv sync
```

## Usage

```bash
# 1. Train the feature extractors (pendigits uses raw features, no extractor)
python -m src.train.speech_extractor
python -m src.train.har_extractor

# 2. Run the adapter comparison sweep for a config
python -m src.compare speech_baseline
python -m src.compare har_baseline
python -m src.compare pendigits_baseline

# 3. Generate figures
python -m src.compare.export_plots results/sweep_speech_baseline_latest.parquet
```

Available compare configs: `speech_baseline` (Google Speech Commands), `har_baseline` (WISDM-2019), `pendigits_baseline`.

The trained checkpoints (`checkpoints/`) and the result parquets used in the thesis (`results/`) are committed in the repo, so steps 2 and 3 can be run without retraining.
