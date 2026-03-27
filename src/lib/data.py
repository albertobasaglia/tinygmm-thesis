import pathlib
import torch
import torchaudio
import torchaudio.transforms as T
import lightning as L
from torch.utils.data import DataLoader

SAMPLE_RATE = 16_000
N_SAMPLES   = SAMPLE_RATE  # 1 second


class Preprocess:
    """Picklable transform: waveform → normalized mel spectrogram (1, n_mels, time)."""
    def __init__(self, n_mels: int = 40):
        self.mel   = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=320, n_mels=n_mels)
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
    def __init__(self, data_dir: str, n_mels: int = 40, batch_size: int = 64,
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

        if self.held_out_words:
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
    n_mels: int = 40,
) -> torch.Tensor:
    preprocess = Preprocess(n_mels=n_mels)
    raw = torchaudio.datasets.SPEECHCOMMANDS(data_dir, download=False, subset=subset)

    # Filter _walker to only files matching target_class, then truncate to n
    raw._walker = [
        f for f in raw._walker
        if pathlib.Path(f).parent.name == target_class
    ][:n]

    if not raw._walker:
        raise ValueError(f"Class '{target_class}' not found in '{subset}' subset.")

    spectrograms = [preprocess(raw[i][0]) for i in range(len(raw._walker))]
    return torch.stack(spectrograms)