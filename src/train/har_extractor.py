import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lib.models import HARExtractorModule
from lib.data import WISDMDataModule, WISDM_WINDOW_SAMPLES, WISDM_WINDOW_STRIDE, WISDM_CHANNELS

ROOT = Path(__file__).parent.parent.parent

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train a HAR feature extractor on WISDM-2019 (watch accel + gyro)")
parser.add_argument("--embedding_dim", type=int,   default=32,  help="Embedding vector size")
parser.add_argument("--window_samples", type=int,  default=WISDM_WINDOW_SAMPLES,
                    help="Window length in samples (200 = 10 s @ 20 Hz).")
parser.add_argument("--stride",        type=int,   default=WISDM_WINDOW_STRIDE,
                    help="Window stride in samples (100 = 50%% overlap).")
parser.add_argument("--noh",           default=False, action="store_true")
parser.add_argument("--epochs",        type=int,   default=50)
parser.add_argument("--batch_size",    type=int,   default=64)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--patience",      type=int,   default=7,   help="Early stopping patience")
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--data_dir",      type=str,   default=str(ROOT / "data"))
parser.add_argument("--num_workers",   type=int,   default=4)
parser.add_argument("--held_out_subjects", type=int, nargs="+", default=[],
                    help="Subject IDs to exclude from training (e.g. --held_out_subjects 1600 1601)")
parser.add_argument("--resume_from",   type=str, default=None, metavar="CKPT",
                    help="Path to a checkpoint to resume training from")
parser.add_argument("--channels",      type=int, nargs="+", default=[32, 64, 128],
                    help="Conv backbone channel widths (one entry per block).")
parser.add_argument("--dropout",       type=float, default=0.5,
                    help="Dropout probability used in conv blocks and head.")

args = parser.parse_args()

# ── Train ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    L.seed_everything(args.seed, workers=True)

    dm = WISDMDataModule(args.data_dir,
                         window_samples=args.window_samples,
                         stride=args.stride,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         held_out_subjects=args.held_out_subjects,
                         seed=args.seed)
    dm.prepare_data()
    dm.setup()
    print(f"[*] {dm.num_classes} classes | {len(dm.train_ds):,} train / {len(dm.val_ds):,} val")

    sample_batch, _ = next(iter(dm.train_dataloader()))
    print(f"[*] Config  — window_samples={args.window_samples}, stride={args.stride}, "
          f"embedding_dim={args.embedding_dim}, batch_size={args.batch_size}, "
          f"lr={args.lr}, epochs={args.epochs}, seed={args.seed}")
    print(f"[*] Input shape (B, C, T) = {tuple(sample_batch.shape)}")

    if args.held_out_subjects == [] and not args.noh:
        print("No held out subjects. If you are sure about this, use --noh")
        exit(-1)

    if args.held_out_subjects:
        print(f"[*] Held-out subjects (not in training): {args.held_out_subjects}")

    channels = tuple(args.channels)
    module = HARExtractorModule(dm.num_classes, args.embedding_dim, args.lr,
                                held_out_subjects=args.held_out_subjects or None,
                                in_channels=WISDM_CHANNELS,
                                channels=channels,
                                dropout_p=args.dropout)
    module.set_normalization(dm.channel_mean, dm.channel_std)
    print(f"[*] Channels: {channels} | dropout: {args.dropout}")
    print(f"[*] Per-channel mean: {dm.channel_mean.tolist()}")
    print(f"[*] Per-channel std:  {dm.channel_std.tolist()}")
    print(f"[*] Parameters: {sum(p.numel() for p in module.parameters()):,}")

    ch_str = "-".join(map(str, channels))
    ckpt_name = f"har_extractor_ch{ch_str}_emb{args.embedding_dim}_dp{args.dropout}_seed{args.seed}"
    logger = CSVLogger("logs", name="har_extractor")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience, verbose=True),
            ModelCheckpoint(monitor="val_loss", filename=ckpt_name, save_top_k=1, mode="min", save_last=True),
        ],
        deterministic=True,
        logger=logger,
    )
    trainer.fit(module, datamodule=dm, ckpt_path=args.resume_from)
    trainer.test(module, datamodule=dm, ckpt_path="best")

    print(f"\nBest checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print("Load with: HARExtractorModule.load_from_checkpoint(path)")
