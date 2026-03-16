import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lib.models import SpeechExtractorModule
from lib.data import SpeechCommandsDataModule

ROOT = Path(__file__).parent.parent.parent

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train a speech feature extractor on Google Speech Commands")
parser.add_argument("--embedding_dim", type=int,   default=32,  help="Embedding vector size")
parser.add_argument("--n_mels",        type=int,   default=64,  help="Number of mel filterbanks")
parser.add_argument("--epochs",        type=int,   default=50)
parser.add_argument("--batch_size",    type=int,   default=64)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--patience",      type=int,   default=7,   help="Early stopping patience")
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--data_dir",      type=str,   default=str(ROOT / "data"))
parser.add_argument("--num_workers",   type=int,   default=4)
parser.add_argument("--held_out_words", type=str,  nargs="+", default=[],
                    help="Word classes to exclude from training (e.g. --held_out_words yes no wow)")
args = parser.parse_args()

# ── Train ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    L.seed_everything(args.seed, workers=True)

    dm = SpeechCommandsDataModule(args.data_dir, args.n_mels, args.batch_size, args.num_workers,
                                  held_out_words=args.held_out_words)
    dm.prepare_data()
    dm.setup()
    print(f"[*] {dm.num_classes} classes | {len(dm.train_ds):,} train / {len(dm.val_ds):,} val / {len(dm.test_ds):,} test")

    if args.held_out_words:
        print(f"[*] Held-out words (not in training): {args.held_out_words}")
    module = SpeechExtractorModule(dm.num_classes, args.embedding_dim, args.lr,
                                   held_out_words=args.held_out_words or None)
    print(f"[*] Parameters: {sum(p.numel() for p in module.parameters()):,}")

    ckpt_name = f"speech_extractor_emb{args.embedding_dim}_seed{args.seed}"
    logger = CSVLogger("logs", name="speech_extractor")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience, verbose=True),
            ModelCheckpoint(monitor="val_loss", filename=ckpt_name, save_top_k=1, mode="min"),
        ],
        deterministic=True,
        logger=logger
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")

    print(f"\nBest checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print("Load with: SpeechExtractorModule.load_from_checkpoint(path)")
