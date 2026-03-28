import torch
import torch.nn as nn
import lightning as L


class SpeechFeatureExtractor(nn.Module):
    """
    CNN trained on mel spectrograms of Google Speech Commands.
    Backbone: conv stack → AdaptiveAvgPool → flat 128-D vector.
    Bottleneck: 128 → (embedding_dim*2) → embedding_dim  (configurable).
    Head: linear classifier, bypassed via return_embedding=True.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        inter_dim = max(embedding_dim * 2, 64)

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128,3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(128, inter_dim), nn.ReLU(),
            nn.Linear(inter_dim, embedding_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.embedding_head(self.backbone(x))
        return embedding if return_embedding else self.classifier(embedding)


class SpeechExtractorModule(L.LightningModule):
    def __init__(self, num_classes: int, embedding_dim: int, lr: float,
                 held_out_words: list = None):
        super().__init__()
        self.save_hyperparameters()
        self.model     = SpeechFeatureExtractor(num_classes, embedding_dim)
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
    Small autoencoder.
    """
    def __init__(self, input_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SpeechAnomalyModule(L.LightningModule):
    """
    Trainer for the Autoencoder. 
    Focuses on Reconstruction Loss (MSE) for One-Class Classification.
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
