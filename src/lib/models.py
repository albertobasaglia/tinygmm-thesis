import numpy as np
import torch
import torch.nn as nn
import lightning as L


class SpeechFeatureExtractor(nn.Module):
    """
    CNN trained on mel spectrograms of Google Speech Commands.
    Backbone: conv stack (one block per channel in `channels`) -> AdaptiveAvgPool -> flat vector.
    Bottleneck: channels[-1] -> (embedding_dim*2) -> embedding_dim.
    Head: linear classifier, bypassed via return_embedding=True.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 32,
                 channels: tuple = (32, 64, 128), dropout_p: float = 0.5):
        super().__init__()
        inter_dim = max(embedding_dim * 2, 64)

        layers = []
        prev = 1
        for i, ch in enumerate(channels):
            layers += [nn.Conv2d(prev, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU()]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout2d(dropout_p))
            prev = ch
        layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(dropout_p), nn.Flatten()]
        self.backbone = nn.Sequential(*layers)

        self.embedding_head = nn.Sequential(
            nn.Linear(channels[-1], inter_dim), nn.ReLU(),
            nn.Linear(inter_dim, embedding_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.embedding_head(self.backbone(x))
        return embedding if return_embedding else self.classifier(embedding)


class SpeechExtractorModule(L.LightningModule):
    def __init__(self, num_classes: int, embedding_dim: int, lr: float,
                 held_out_words: list = None, channels: tuple = (32, 64, 128),
                 dropout_p: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model     = SpeechFeatureExtractor(num_classes, embedding_dim,
                                                channels=tuple(channels),
                                                dropout_p=dropout_p)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        return self.model(x, return_embedding)

    def _shared_step(self, batch):
        specs, labels = batch
        out  = self.model(specs)
        loss = self.criterion(out, labels)
        acc  = (out.argmax(1) == labels).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def test_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class HARFeatureExtractor(nn.Module):
    """
    1D-CNN trained on (channels, time) sensor windows from WISDM-2019.
    1D analog of SpeechFeatureExtractor: same conv-block / bottleneck / head shape.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 32, in_channels: int = 6,
                 channels: tuple = (32, 64, 128), dropout_p: float = 0.5):
        super().__init__()
        inter_dim = max(embedding_dim * 2, 64)

        layers = []
        prev = in_channels
        for i, ch in enumerate(channels):
            kernel = 5 if i < len(channels) - 1 else 3
            padding = kernel // 2
            layers += [nn.Conv1d(prev, ch, kernel, padding=padding),
                       nn.BatchNorm1d(ch), nn.ReLU()]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout1d(dropout_p))
            prev = ch
        layers += [nn.AdaptiveAvgPool1d(1), nn.Dropout(dropout_p), nn.Flatten()]
        self.backbone = nn.Sequential(*layers)

        self.embedding_head = nn.Sequential(
            nn.Linear(channels[-1], inter_dim), nn.ReLU(),
            nn.Linear(inter_dim, embedding_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.embedding_head(self.backbone(x))
        return embedding if return_embedding else self.classifier(embedding)


class HARExtractorModule(L.LightningModule):
    """Lightning wrapper for HARFeatureExtractor.

    Per-channel normalization (mean/std) lives in buffers so it travels with the
    checkpoint; the compare-time provider does not need to know the stats.
    Call `set_normalization(mean, std)` once after construction (the train script
    does this with the DataModule's training-set statistics).
    """
    def __init__(self, num_classes: int, embedding_dim: int, lr: float,
                 held_out_subjects: list = None, in_channels: int = 6,
                 channels: tuple = (32, 64, 128), dropout_p: float = 0.5,
                 optimizer: str = "adamw", weight_decay: float = 1e-4,
                 lr_patience: int = 3, lr_factor: float = 0.5, lr_min: float = 1e-6):
        super().__init__()
        self.save_hyperparameters()
        self.model = HARFeatureExtractor(num_classes, embedding_dim,
                                         in_channels=in_channels,
                                         channels=tuple(channels),
                                         dropout_p=dropout_p)
        self.criterion = nn.CrossEntropyLoss()
        self.register_buffer("channel_mean", torch.zeros(in_channels))
        self.register_buffer("channel_std", torch.ones(in_channels))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        if mean_t.shape != self.channel_mean.shape:
            raise ValueError(
                f"mean shape {tuple(mean_t.shape)} != expected {tuple(self.channel_mean.shape)}"
            )
        self.channel_mean.copy_(mean_t)
        self.channel_std.copy_(std_t)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return (x - self.channel_mean.view(1, -1, 1)) / self.channel_std.view(1, -1, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        return self.model(self._normalize(x), return_embedding)

    def _shared_step(self, batch):
        x, labels = batch
        out = self.forward(x)
        loss = self.criterion(out, labels)
        acc = (out.argmax(1) == labels).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc},
                      on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def test_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})

    def configure_optimizers(self):
        opt_name = self.hparams.optimizer.lower()
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer!r}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.hparams.lr_patience,
            factor=self.hparams.lr_factor, min_lr=self.hparams.lr_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class SpeechAutoencoder(nn.Module):
    """
    Bottleneck-style autoencoder for 1-class classification.
    Input: 32-D embedding from SpeechFeatureExtractor.
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SmallAutoencoder(nn.Module):
    """
    Small Denoising Autoencoder.
    dropout_p=0.0 disables denoising (standard AE behaviour).
    """
    def __init__(self, input_dim: int = 32, latent_dim: int = 8, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(self.dropout(x)))


class SpeechAnomalyModule(L.LightningModule):
    """
    Trainer for the Autoencoder. 
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, latent_dim: int = 8, 
                 lr: float = 1e-3, noise_std: float = 0.02):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpeechAutoencoder(input_dim, hidden_dim, latent_dim)
        self.criterion = nn.MSELoss()

        self.register_buffer("computed_threshold", torch.tensor(0.0))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x = batch[0] if isinstance(batch, list) else batch

        reconstructed = self.model(x)
        loss = self.criterion(reconstructed, x)
        
        self.log("train_reconst_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        reconstructed = self.model(x)
        loss = self.criterion(reconstructed, x)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

    @torch.no_grad()
    def get_anomaly_score(self, x: torch.Tensor):
        reconstructed = self.model(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)
