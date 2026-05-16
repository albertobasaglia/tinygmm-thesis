import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lib.models import SpeechExtractorModule
from lib.data import SpeechCommandsDataModule, N_MELS

ROOT = Path(__file__).parent.parent.parent

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train a speech feature extractor on Google Speech Commands")
parser.add_argument("--embedding_dim", type=int,   default=32,  help="Embedding vector size")
parser.add_argument("--n_mels",        type=int,   default=N_MELS,  help="Number of mel filterbanks")
parser.add_argument("--noh",  default=False,
                    action="store_true")
parser.add_argument("--epochs",        type=int,   default=50)
parser.add_argument("--batch_size",    type=int,   default=64)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--patience",      type=int,   default=7,   help="Early stopping patience")
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--data_dir",      type=str,   default=str(ROOT / "data"))
parser.add_argument("--num_workers",   type=int,   default=4)
parser.add_argument("--held_out_words", type=str,  nargs="+", default=[],
                    help="Word classes to exclude from training (e.g. --held_out_words yes no wow)")
parser.add_argument("--resume_from", type=str, default=None, metavar="CKPT",
                    help="Path to a checkpoint to resume training from (restores epoch, optimizer, LR scheduler)")
parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128],
                    help="Conv backbone channel widths (one entry per block). Default reproduces the original architecture.")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout probability used in conv blocks and head. 0.0 disables dropout (useful for tiny channels).")

args = parser.parse_args()

# ── Train ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    L.seed_everything(args.seed, workers=True)

    dm = SpeechCommandsDataModule(args.data_dir, args.n_mels, args.batch_size, args.num_workers,
                                  held_out_words=args.held_out_words)
    dm.prepare_data()
    dm.setup()
    print(f"[*] {dm.num_classes} classes | {len(dm.train_ds):,} train / {len(dm.val_ds):,} val / {len(dm.test_ds):,} test")

    sample_batch, _ = next(iter(dm.train_dataloader()))
    print(f"[*] Config  — n_mels={args.n_mels}, embedding_dim={args.embedding_dim}, "
          f"batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}, seed={args.seed}")
    print(f"[*] Input shape (B, C, mels, time) = {tuple(sample_batch.shape)}")

    if args.held_out_words == [] and not args.noh:
        print("No held out words. If you are sure about this, use --noh")
        exit(-1)

    if args.held_out_words:
        print(f"[*] Held-out words (not in training): {args.held_out_words}")
    channels = tuple(args.channels)
    module = SpeechExtractorModule(dm.num_classes, args.embedding_dim, args.lr,
                                   held_out_words=args.held_out_words or None,
                                   channels=channels,
                                   dropout_p=args.dropout)
    print(f"[*] Channels: {channels} | dropout: {args.dropout}")
    print(f"[*] Parameters: {sum(p.numel() for p in module.parameters()):,}")

    ch_str = "-".join(map(str, channels))
    ckpt_name = f"speech_extractor_ch{ch_str}_emb{args.embedding_dim}_dp{args.dropout}_seed{args.seed}-{{epoch:02d}}-{{val_loss:.4f}}"
    logger = CSVLogger("logs", name="speech_extractor")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience, verbose=True),
            ModelCheckpoint(monitor="val_loss", filename=ckpt_name, save_top_k=3, mode="min", save_last=True),
        ],
        deterministic=True,
        logger=logger
    )
    trainer.fit(module, datamodule=dm, ckpt_path=args.resume_from)
    trainer.test(module, datamodule=dm, ckpt_path="best")

    print(f"\nBest checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print("Load with: SpeechExtractorModule.load_from_checkpoint(path)")
