import logging
import pathlib
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import lightning as L
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
N_SAMPLES   = SAMPLE_RATE  # 1 second
N_MELS      = 40
HOP_LENGTH  = 320

# WISDM-2019 constants
WISDM_URL = (
    "https://archive.ics.uci.edu/static/public/507/"
    "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
)
WISDM_DIR = "wisdm-2019"
WISDM_SAMPLE_RATE_HZ = 20
WISDM_BUCKET_NS = 1_000_000_000 // WISDM_SAMPLE_RATE_HZ  # 50 ms
WISDM_WINDOW_SAMPLES = 200  # 10 s @ 20 Hz
WISDM_WINDOW_STRIDE = 100   # 50% overlap
WISDM_CHANNELS = 6  # ax, ay, az, gx, gy, gz


class Preprocess:
    """Picklable transform: waveform → normalized mel spectrogram (1, n_mels, time)."""
    def __init__(self, n_mels: int = N_MELS):
        self.mel   = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=HOP_LENGTH, n_mels=n_mels)
        self.to_db = T.AmplitudeToDB()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        n = waveform.shape[1]
        if n < N_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - n))
        else:
            waveform = waveform[:, :N_SAMPLES]
        spec = self.to_db(self.mel(waveform))
        return (spec - spec.mean()) / (spec.std() + 1e-8)


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, label_to_idx: dict, preprocess: Preprocess):
        self.data         = raw_dataset
        self.label_to_idx = label_to_idx
        self.preprocess   = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, _, label, *_ = self.data[idx]
        return self.preprocess(waveform), self.label_to_idx[label]


class SpeechCommandsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, n_mels: int = N_MELS, batch_size: int = 64,
                 num_workers: int = 4, held_out_words: list = None):
        super().__init__()
        self.data_dir       = data_dir
        self.n_mels         = n_mels
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.held_out_words = set(held_out_words or [])
        self.preprocess     = Preprocess(n_mels)
        self.label_to_idx: dict = {}
        self.num_classes: int   = 0

    def prepare_data(self):
        for subset in ("training", "validation", "testing"):
            torchaudio.datasets.SPEECHCOMMANDS(self.data_dir, download=True, subset=subset)

    def setup(self, stage=None):
        train_raw = torchaudio.datasets.SPEECHCOMMANDS(self.data_dir, download=False, subset="training")
        val_raw   = torchaudio.datasets.SPEECHCOMMANDS(self.data_dir, download=False, subset="validation")
        test_raw  = torchaudio.datasets.SPEECHCOMMANDS(self.data_dir, download=False, subset="testing")

        all_labels_in_dataset = sorted({pathlib.Path(f).parent.name for f in train_raw._walker})

        if self.held_out_words:
            unknown = self.held_out_words - set(all_labels_in_dataset)
            if unknown:
                raise ValueError(
                    f"Unknown held-out word(s): {sorted(unknown)}\n"
                    f"Valid classes: {all_labels_in_dataset}"
                )
            for raw in (train_raw, val_raw, test_raw):
                raw._walker = [f for f in raw._walker
                               if pathlib.Path(f).parent.name not in self.held_out_words]

        all_labels        = sorted({pathlib.Path(f).parent.name for f in train_raw._walker})
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        self.num_classes  = len(all_labels)

        self.train_ds = SpeechCommandsDataset(train_raw, self.label_to_idx, self.preprocess)
        self.val_ds   = SpeechCommandsDataset(val_raw,   self.label_to_idx, self.preprocess)
        self.test_ds  = SpeechCommandsDataset(test_raw,  self.label_to_idx, self.preprocess)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def train_dataloader(self): return self._loader(self.train_ds, shuffle=True)
    def val_dataloader(self):   return self._loader(self.val_ds,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.test_ds,  shuffle=False)

def get_spectrograms(
    data_dir: str,
    target_class: str,
    n: int = 100,
    subset: str = "training",
    n_mels: int = N_MELS,
) -> torch.Tensor:
    preprocess = Preprocess(n_mels=n_mels)
    raw = torchaudio.datasets.SPEECHCOMMANDS(data_dir, download=True, subset=subset)

    # Filter _walker to only files matching target_class, then truncate to n
    raw._walker = [
        f for f in raw._walker
        if pathlib.Path(f).parent.name == target_class
    ][:n]

    if not raw._walker:
        raise ValueError(f"Class '{target_class}' not found in '{subset}' subset.")

    spectrograms = [preprocess(raw[i][0]) for i in range(len(raw._walker))]
    return torch.stack(spectrograms)


# ──────────────────────────────────────────────────────────────────────────────
# WISDM-2019 (Human Activity Recognition)
# ──────────────────────────────────────────────────────────────────────────────


def _wisdm_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / WISDM_DIR


def _find_wisdm_watch_dir(root: Path) -> Path | None:
    """Find the `raw/watch` directory anywhere under root, since the archive's
    top-level folder name has varied between UCI mirrors / re-uploads."""
    for accel in root.rglob("accel"):
        watch = accel.parent
        if watch.name == "watch" and (watch / "gyro").is_dir():
            return watch
    return None


def _wisdm_aligned_parquet(data_dir: str | Path) -> Path:
    return _wisdm_root(data_dir) / "watch_aligned.parquet"


def _download_wisdm(data_dir: str | Path) -> None:
    """Download and unzip WISDM-2019 if the raw watch directory isn't present.

    Uses curl + system `unzip` because the archive is ZIP64 and Python's stdlib
    zipfile module raises EOFError mid-extraction on some members.
    """
    root = _wisdm_root(data_dir)
    if _find_wisdm_watch_dir(root) is not None:
        return
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "wisdm-dataset.zip"

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        log.info("Downloading WISDM-2019 from %s", WISDM_URL)
        if shutil.which("curl"):
            subprocess.run(
                ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "2",
                 "-o", str(zip_path), WISDM_URL],
                check=True,
            )
        else:
            urllib.request.urlretrieve(WISDM_URL, zip_path)
        if zip_path.stat().st_size < 1_000_000:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"WISDM download produced only {zip_path.stat().st_size if zip_path.exists() else 0} bytes; "
                f"expected ~300MB. Check network or URL."
            )

    log.info("Unpacking %s", zip_path)
    if not shutil.which("unzip"):
        raise RuntimeError(
            "System `unzip` not found on PATH. Install it (e.g. `brew install unzip`) "
            "and retry; Python's stdlib zipfile fails on this ZIP64 archive."
        )
    subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(root)],
        check=True,
    )
    if _find_wisdm_watch_dir(root) is None:
        layout = "\n".join(
            f"  {p.relative_to(root)}" for p in sorted(root.rglob("*"))
            if p.is_dir() and len(p.relative_to(root).parts) <= 3
        )
        raise RuntimeError(
            f"After unpacking, could not find a `raw/watch` directory under {root}.\n"
            f"Top-level layout:\n{layout}"
        )


def _parse_wisdm_file(path: Path, prefix: str) -> pd.DataFrame:
    """Parse one raw WISDM file. prefix is 'a' (accel) or 'g' (gyro).

    Each line: '<subject>,<activity>,<timestamp_ns>,<x>,<y>,<z>;'.
    """
    cols = ["subject", "activity", "t_ns", f"{prefix}x", f"{prefix}y", f"{prefix}z"]
    df = pd.read_csv(
        path, header=None, names=cols,
        sep=",", engine="c", skip_blank_lines=True,
        dtype=str, on_bad_lines="skip",
    )
    df[f"{prefix}z"] = df[f"{prefix}z"].str.rstrip(";")
    df = df.dropna()
    df = df[df["t_ns"].str.isdigit()]
    df["subject"] = df["subject"].astype(np.int64)
    df["t_ns"] = df["t_ns"].astype(np.int64)
    for c in (f"{prefix}x", f"{prefix}y", f"{prefix}z"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df.dropna()
    return df


def _align_wisdm(data_dir: str | Path) -> pd.DataFrame:
    """Parse all watch accel + gyro files and time-align into one parquet (cached).

    Returns a frame with columns
    [subject, activity, bucket, t_ns, ax, ay, az, gx, gy, gz],
    sorted by (subject, activity, bucket).
    """
    cache = _wisdm_aligned_parquet(data_dir)
    if cache.exists():
        return pd.read_parquet(cache)

    _download_wisdm(data_dir)
    root = _find_wisdm_watch_dir(_wisdm_root(data_dir))
    if root is None:
        raise RuntimeError(
            f"No `raw/watch` directory under {_wisdm_root(data_dir)}. "
            f"Did download/unpack succeed?"
        )
    accel_files = sorted((root / "accel").glob("data_*_accel_watch.txt"))
    gyro_files = sorted((root / "gyro").glob("data_*_gyro_watch.txt"))
    if not accel_files or not gyro_files:
        raise RuntimeError(
            f"No WISDM watch files found under {root}. Did download/unpack succeed?"
        )

    log.info("Parsing %d accel + %d gyro WISDM files",
             len(accel_files), len(gyro_files))
    accel = pd.concat([_parse_wisdm_file(p, "a") for p in accel_files],
                      ignore_index=True)
    gyro = pd.concat([_parse_wisdm_file(p, "g") for p in gyro_files],
                     ignore_index=True)

    accel["bucket"] = (accel["t_ns"] + WISDM_BUCKET_NS // 2) // WISDM_BUCKET_NS
    gyro["bucket"] = (gyro["t_ns"] + WISDM_BUCKET_NS // 2) // WISDM_BUCKET_NS
    accel = accel.groupby(["subject", "activity", "bucket"], as_index=False)[
        ["ax", "ay", "az"]].mean()
    gyro = gyro.groupby(["subject", "activity", "bucket"], as_index=False)[
        ["gx", "gy", "gz"]].mean()

    aligned = accel.merge(gyro, on=["subject", "activity", "bucket"], how="inner")
    aligned["t_ns"] = aligned["bucket"] * WISDM_BUCKET_NS
    aligned = aligned.sort_values(["subject", "activity", "bucket"]).reset_index(drop=True)

    n_subjects = aligned["subject"].nunique()
    n_dropped = (
        set(accel["subject"].unique()) | set(gyro["subject"].unique())
    ) - set(aligned["subject"].unique())
    log.info("WISDM aligned: %d samples, %d subjects (dropped %d for missing accel/gyro overlap)",
             len(aligned), n_subjects, len(n_dropped))

    cache.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(cache, index=False)
    return aligned


def _build_windows(df: pd.DataFrame, window_samples: int, stride: int):
    """Slide windows over each (subject, activity) chunk, splitting on time gaps.

    Returns a list of (subject, activity, ndarray(window_samples, 6)) tuples.
    """
    cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    out = []
    for (subject, activity), group in df.groupby(["subject", "activity"], sort=False):
        buckets = group["bucket"].to_numpy()
        if len(buckets) < window_samples:
            continue
        arr = group[cols].to_numpy(dtype=np.float32)

        gaps = np.where(np.diff(buckets) > 1)[0] + 1
        chunk_starts = np.concatenate([[0], gaps]).astype(int)
        chunk_ends = np.concatenate([gaps, [len(buckets)]]).astype(int)
        for s, e in zip(chunk_starts, chunk_ends):
            chunk = arr[s:e]
            if len(chunk) < window_samples:
                continue
            for w_start in range(0, len(chunk) - window_samples + 1, stride):
                out.append((int(subject), str(activity),
                            chunk[w_start:w_start + window_samples]))
    return out


class WISDMDataset(torch.utils.data.Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        # windows: (N, 6, T) float32; labels: (N,) int64
        self.windows = torch.from_numpy(windows)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class WISDMDataModule(L.LightningDataModule):
    """Subject-classification DataModule for WISDM-2019 watch (accel + gyro)."""

    def __init__(
        self,
        data_dir: str,
        window_samples: int = WISDM_WINDOW_SAMPLES,
        stride: int = WISDM_WINDOW_STRIDE,
        batch_size: int = 64,
        num_workers: int = 4,
        held_out_subjects: list = None,
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window_samples = window_samples
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.held_out_subjects = set(int(s) for s in (held_out_subjects or []))
        self.val_fraction = val_fraction
        self.seed = seed
        self.label_to_idx: dict = {}
        self.num_classes: int = 0
        self.channel_mean: np.ndarray | None = None
        self.channel_std: np.ndarray | None = None

    def prepare_data(self):
        _download_wisdm(self.data_dir)
        _align_wisdm(self.data_dir)  # warm cache

    def setup(self, stage=None):
        df = _align_wisdm(self.data_dir)
        all_subjects = sorted(df["subject"].unique().tolist())

        unknown = self.held_out_subjects - set(all_subjects)
        if unknown:
            raise ValueError(
                f"Unknown held-out subject(s): {sorted(unknown)}\n"
                f"Valid subjects: {all_subjects}"
            )
        train_subjects = [s for s in all_subjects if s not in self.held_out_subjects]
        if not train_subjects:
            raise ValueError("No training subjects after applying held-out.")

        self.label_to_idx = {s: i for i, s in enumerate(train_subjects)}
        self.num_classes = len(train_subjects)

        train_df = df[df["subject"].isin(train_subjects)]
        windows = _build_windows(train_df, self.window_samples, self.stride)

        rng = np.random.default_rng(self.seed)
        by_group: dict = defaultdict(list)
        for s, a, w in windows:
            by_group[(s, a)].append(w)

        train_X, train_y, val_X, val_y = [], [], [], []
        for (s, a), ws in by_group.items():
            n = len(ws)
            n_val = int(n * self.val_fraction) if n > 1 else 0
            idx = rng.permutation(n)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            cls = self.label_to_idx[s]
            for i in train_idx:
                train_X.append(ws[i]); train_y.append(cls)
            for i in val_idx:
                val_X.append(ws[i]); val_y.append(cls)

        if not train_X:
            raise RuntimeError("Empty training set after windowing/splitting.")

        train_X = np.stack(train_X).astype(np.float32)
        val_X = (np.stack(val_X).astype(np.float32)
                 if val_X else np.zeros((0, self.window_samples, WISDM_CHANNELS),
                                         dtype=np.float32))
        train_y = np.array(train_y, dtype=np.int64)
        val_y = np.array(val_y, dtype=np.int64)

        # Per-channel z-score, computed on training data only
        flat = train_X.reshape(-1, WISDM_CHANNELS)
        self.channel_mean = flat.mean(axis=0).astype(np.float32)
        self.channel_std = (flat.std(axis=0) + 1e-6).astype(np.float32)

        # (N, T, C) -> (N, C, T) for Conv1d
        train_X = train_X.transpose(0, 2, 1)
        val_X = val_X.transpose(0, 2, 1)

        self.train_ds = WISDMDataset(train_X, train_y)
        self.val_ds = WISDMDataset(val_X, val_y)
        self.test_ds = self.val_ds

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers,
                          persistent_workers=(self.num_workers > 0))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds, False)
    def test_dataloader(self):  return self._loader(self.test_ds, False)


def get_windows(
    data_dir: str | Path,
    target_subject: int,
    n: int = 100,
    subset: str = "training",
    window_samples: int = WISDM_WINDOW_SAMPLES,
    stride: int = WISDM_WINDOW_STRIDE,
    seed: int = 0,
) -> torch.Tensor:
    """Return up to n windows of shape (n, 6, window_samples) for one subject.

    Per-subject internal split, stratified by activity:
      subset='training' -> first half of each activity's window stream (enrollment)
      subset='testing'  -> second half (held-out evaluation)
    Windows are then shuffled deterministically and truncated to n.
    """
    if subset not in ("training", "testing"):
        raise ValueError(f"subset must be 'training' or 'testing', got {subset!r}")

    df = _align_wisdm(data_dir)
    sub_df = df[df["subject"] == int(target_subject)]
    if sub_df.empty:
        raise ValueError(f"Subject {target_subject} not found in WISDM dataset.")

    windows = _build_windows(sub_df, window_samples, stride)
    by_act: dict = defaultdict(list)
    for _, a, w in windows:
        by_act[a].append(w)

    selected = []
    for ws in by_act.values():
        mid = len(ws) // 2
        selected.extend(ws[:mid] if subset == "training" else ws[mid:])

    if not selected:
        raise ValueError(
            f"No windows for subject {target_subject} subset {subset!r}."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    selected = selected[:n]

    arr = np.stack(selected).astype(np.float32).transpose(0, 2, 1)  # (n, 6, T)
    return torch.from_numpy(arr)