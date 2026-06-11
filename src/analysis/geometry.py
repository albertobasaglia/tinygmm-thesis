"""
Intrinsic feature-space geometry diagnostics for the one-class thesis claim.

Goal
----
Replace the circular prose ("HAR is less cleanly class-centered than speech,
*because* the GMM beats cosine there") with measured, intrinsic geometry numbers
that DO NOT depend on which adapter won.

For each of the three datasets (speech, har, pendigits) this script reuses the
SAME embedding providers the comparison pipeline uses, pulls the held-out TEST
target-class embeddings (the exact vectors the headline numbers are scored on),
and computes, per class then averaged across classes:

  1. Within-class ANGULAR CONCENTRATION
       mean cosine similarity of each class sample to its class-mean direction.
       High  -> the class lives near one direction (cosine-prototype regime).

  2. Within-class COVARIANCE STRUCTURE
       - leading-eigenvalue share (lambda_1 / trace) and effective rank of the
         within-class covariance: how anisotropic the cloud is.
       - off-diagonal correlation mass of the within-class CORRELATION matrix
         (mean |off-diag| and off-diag Frobenius mass ratio): exactly what a full
         covariance models and a diagonal one cannot.

  3. MAGNITUDE-vs-DIRECTION separability (the non-circular pendigits clincher)
       a Fisher-style between/within scatter ratio computed on
         (a) raw vectors,
         (b) unit-normalised vectors (direction only),
         (c) per-sample norms alone (magnitude only).
       If normalising barely hurts -> direction carries the separability
       (cosine should work). If normalising destroys it -> direction is
       uninformative and magnitude carries it (cosine should collapse).

Run:
    python -m src.analysis.geometry          # all three datasets, test split
    python -m src.analysis.geometry --split validation
    python -m src.analysis.geometry --datasets speech har

It is read-only on everything except this file: it only calls the existing
providers and reads checkpoints / class lists already on disk.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from embeddings.speech import SpeechEmbeddingProvider
from embeddings.har import HAREmbeddingProvider
from embeddings.tabular import TabularEmbeddingProvider

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
CLASSES_DIR = ROOT / "src" / "compare" / "classes"

# Mirror the comparison pipeline constants (src/compare/__main__.py).
TEST_N = 500          # held-out samples per class returned by get_embeddings
TRAIN_N_MAX = 195     # max(TRAIN_N) the pipeline passes as train_n


def read_classes(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Per-class geometric statistics (all intrinsic, adapter-free)
# ──────────────────────────────────────────────────────────────────────────────


def angular_concentration(X: np.ndarray) -> tuple[float, float]:
    """Mean (and std) cosine similarity of each sample to the class-mean DIRECTION.

    The class-mean direction is mean(X) normalised. Returns (mean, std) over
    samples. 1.0 => perfectly ray-like cluster.
    """
    mu = X.mean(axis=0)
    mu_norm = np.linalg.norm(mu)
    if mu_norm < 1e-12:
        return 0.0, 0.0
    mu_dir = mu / mu_norm
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 1e-12
    cos = (X[keep] @ mu_dir) / norms[keep]
    return float(cos.mean()), float(cos.std())


def covariance_structure(X: np.ndarray) -> dict:
    """Anisotropy + off-diagonal correlation mass of the within-class covariance."""
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    trace = float(eigvals.sum())

    # Anisotropy: leading-eigenvalue share and effective (participation) rank.
    lead_share = float(eigvals.max() / trace) if trace > 0 else float("nan")
    p = eigvals / trace if trace > 0 else np.zeros_like(eigvals)
    eff_rank = float(np.exp(-np.sum(np.where(p > 0, p * np.log(p), 0.0))))  # entropy rank

    # Off-diagonal correlation mass from the CORRELATION matrix.
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    D = corr.shape[0]
    off_mask = ~np.eye(D, dtype=bool)
    mean_abs_offdiag = float(np.abs(corr[off_mask]).mean())
    # Frobenius mass ratio: ||offdiag||_F^2 / ||corr||_F^2.
    offdiag_frob = float((corr[off_mask] ** 2).sum())
    total_frob = float((corr ** 2).sum())
    offdiag_frac = offdiag_frob / total_frob if total_frob > 0 else float("nan")

    return {
        "lead_eig_share": lead_share,
        "eff_rank": eff_rank,
        "dim": float(D),
        "mean_abs_offdiag_corr": mean_abs_offdiag,
        "offdiag_frob_frac": offdiag_frac,
    }


def within_class_correlation(X: np.ndarray) -> np.ndarray:
    """The DxD within-class correlation matrix (covariance scaled to unit diagonal)."""
    cov = np.atleast_2d(np.cov(X, rowvar=False))
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return cov / np.outer(std, std)


def fisher_ratio(class_arrays: list[np.ndarray]) -> float:
    """Multiclass Fisher-style separability: trace(S_b) / trace(S_w).

    S_w = average within-class covariance (pooled).
    S_b = covariance of the class means (between-class scatter).
    Higher => classes more separable in this representation.
    Defined on whatever feature space `class_arrays` already live in.
    """
    means = np.stack([Xc.mean(axis=0) for Xc in class_arrays])
    grand = np.concatenate(class_arrays, axis=0).mean(axis=0)

    # Within: pooled within-class scatter (trace = sum of per-axis within variance).
    sw = 0.0
    n_total = 0
    for Xc in class_arrays:
        d = Xc - Xc.mean(axis=0)
        sw += float((d ** 2).sum())  # trace of class scatter (sum over axes & samples)
        n_total += len(Xc)
    sw /= n_total

    # Between: scatter of class means about the grand mean, weighted by class size.
    sb = 0.0
    for Xc, m in zip(class_arrays, means):
        sb += len(Xc) * float(((m - grand) ** 2).sum())
    sb /= n_total

    return sb / sw if sw > 0 else float("nan")


def fisher_1d(class_arrays_1d: list[np.ndarray]) -> float:
    """Fisher ratio for a scalar feature (per-sample norm). Same definition, 1-D."""
    return fisher_ratio([a.reshape(-1, 1) for a in class_arrays_1d])


# ──────────────────────────────────────────────────────────────────────────────
# Provider wiring — mirrors src/compare/__main__.py --split test
# ──────────────────────────────────────────────────────────────────────────────


def build_providers(dataset: str, split: str, device: str):
    """Return a list of (target_label, provider) for the requested split.

    split is 'test' (held-out test classes; headline) or 'validation' (the
    pipeline's val classes: held_out minus test).
    """
    if dataset == "speech":
        ckpt = ROOT / "checkpoints" / "speech" / "best.ckpt"
        meta = torch.load(ckpt, weights_only=True, map_location="cpu")
        held_out = list(meta["hyper_parameters"].get("held_out_words") or [])
        edim = int(meta["hyper_parameters"]["embedding_dim"])
        test_words = set(read_classes(CLASSES_DIR / "speech" / "test.txt"))
        if split == "test":
            targets = sorted(test_words)
        else:
            targets = [w for w in held_out if w not in test_words]
        return edim, [
            (
                w,
                SpeechEmbeddingProvider(
                    ckpt, edim, ROOT / "data",
                    target_class=w,
                    other_classes=[o for o in targets if o != w],
                    device=device,
                ),
            )
            for w in targets
        ]

    if dataset == "har":
        ckpt = ROOT / "checkpoints" / "har" / "best.ckpt"
        meta = torch.load(ckpt, weights_only=True, map_location="cpu")
        held_out = [int(s) for s in (meta["hyper_parameters"].get("held_out_subjects") or [])]
        edim = int(meta["hyper_parameters"]["embedding_dim"])
        test_subj = {int(s) for s in read_classes(CLASSES_DIR / "har" / "test.txt")}
        if split == "test":
            targets = sorted(test_subj)
        else:
            targets = [s for s in held_out if s not in test_subj]
        return edim, [
            (
                s,
                HAREmbeddingProvider(
                    ckpt, edim, ROOT / "data",
                    target_class=s,
                    other_classes=[o for o in targets if o != s],
                    device=device,
                ),
            )
            for s in targets
        ]

    if dataset == "pendigits":
        from lib.data import download_pendigits

        all_digits = [str(i) for i in range(10)]
        test_digits = read_classes(CLASSES_DIR / "pendigits" / "test.txt")
        if split == "test":
            targets = list(test_digits)
        else:
            targets = [d for d in all_digits if d not in set(test_digits)]
        data_path = download_pendigits(ROOT / "data")
        edim = 16
        return edim, [
            (
                d,
                TabularEmbeddingProvider(
                    data_path=data_path,
                    label_column="class",
                    target_class=d,
                    other_classes=[o for o in targets if o != d],
                ),
            )
            for d in targets
        ]

    raise ValueError(f"unknown dataset {dataset!r}")


def gather_test_embeddings(dataset: str, split: str, device: str):
    """Return (edim, targets, test_targets) where test_targets[label] is the
    (TEST_N, dim) held-out array the headline numbers were scored on.

    These are EXACTLY the `test_target` arrays the pipeline feeds to the
    adapters' score() for the final test. For pendigits they carry the same
    per-class StandardScaler standardisation the pipeline applies.
    """
    edim, providers = build_providers(dataset, split, device)
    test_targets: dict = {}
    for label, prov in providers:
        train_emb, test_target, _test_other = prov.get_embeddings(TRAIN_N_MAX, TEST_N)
        test_targets[label] = np.asarray(test_target, dtype=np.float64)
        log.info("%s/%s class=%s -> test_target %s", dataset, split, label, test_target.shape)
    return edim, list(test_targets.keys()), test_targets


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────


def analyse_dataset(dataset: str, split: str, device: str) -> dict:
    edim, labels, test_targets = gather_test_embeddings(dataset, split, device)

    per_class = []
    for lbl in labels:
        X = test_targets[lbl]
        cos_mean, cos_std = angular_concentration(X)
        cov = covariance_structure(X)
        per_class.append({"label": lbl, "cos_mean": cos_mean, "cos_std": cos_std, **cov})

    def avg(key):
        return float(np.mean([pc[key] for pc in per_class]))

    def spread(key):
        return float(np.std([pc[key] for pc in per_class]))

    # ----- Magnitude vs direction separability (Fisher), across the test classes.
    raw = [test_targets[l] for l in labels]
    unit = [X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None) for X in raw]
    norms = [np.linalg.norm(X, axis=1) for X in raw]

    fisher_raw = fisher_ratio(raw)
    fisher_dir = fisher_ratio(unit)
    fisher_mag = fisher_1d(norms)
    # Fraction of raw separability retained by direction only.
    dir_retention = fisher_dir / fisher_raw if fisher_raw > 0 else float("nan")

    # Averaged within-class correlation matrix (for the heatmap figure).
    mean_corr = np.mean([within_class_correlation(X) for X in raw], axis=0)

    return {
        "_mean_corr": mean_corr,
        "dataset": dataset,
        "split": split,
        "n_classes": len(labels),
        "dim": edim,
        # 1. angular concentration
        "cos_mean": avg("cos_mean"),
        "cos_mean_across_class_std": spread("cos_mean"),
        "cos_within_std": avg("cos_std"),
        # 2. covariance structure
        "lead_eig_share": avg("lead_eig_share"),
        "eff_rank": avg("eff_rank"),
        "mean_abs_offdiag_corr": avg("mean_abs_offdiag_corr"),
        "offdiag_frob_frac": avg("offdiag_frob_frac"),
        # 3. magnitude vs direction
        "fisher_raw": fisher_raw,
        "fisher_dir": fisher_dir,
        "fisher_mag": fisher_mag,
        "dir_retention": dir_retention,
        "_per_class": per_class,
    }


def print_table(results: list[dict]):
    cols = [
        ("dataset", "{:>9}", "dataset"),
        ("n_classes", "{:>3d}", "K"),
        ("dim", "{:>3d}", "D"),
        ("cos_mean", "{:>7.3f}", "cosMean"),
        ("cos_mean_across_class_std", "{:>7.3f}", "cosM_sd"),
        ("cos_within_std", "{:>7.3f}", "cosWsd"),
        ("lead_eig_share", "{:>7.3f}", "lead_ev"),
        ("eff_rank", "{:>7.2f}", "effRank"),
        ("mean_abs_offdiag_corr", "{:>7.3f}", "offdiag"),
        ("offdiag_frob_frac", "{:>7.3f}", "offFrob"),
        ("fisher_raw", "{:>8.3f}", "F_raw"),
        ("fisher_dir", "{:>8.3f}", "F_dir"),
        ("fisher_mag", "{:>8.3f}", "F_mag"),
        ("dir_retention", "{:>7.2f}", "dirRet"),
    ]
    header = " ".join(f"{h:>{max(7, len(h))}}" for _k, _f, h in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        cells = []
        for k, fmt, h in cols:
            v = r[k]
            cells.append(fmt.format(v))
        print(" ".join(f"{c:>{max(7, len(h))}}" for c, (_, __, h) in zip(cells, cols)))

    print(
        "\nLegend:"
        "\n  cosMean  mean cosine sim of class samples to class-mean direction (angular concentration)"
        "\n  cosM_sd  std of cosMean ACROSS classes;  cosWsd  mean WITHIN-class std of those cosines"
        "\n  lead_ev  leading-eigenvalue share of within-class covariance (anisotropy, higher=more)"
        "\n  effRank  entropy effective rank of within-class covariance (out of D)"
        "\n  offdiag  mean |off-diagonal| of within-class CORRELATION matrix"
        "\n  offFrob  off-diagonal Frobenius mass fraction of the correlation matrix"
        "\n  F_raw/F_dir/F_mag  Fisher trace(S_b)/trace(S_w) on raw / unit-normalised / norm-only features"
        "\n  dirRet   F_dir / F_raw : fraction of class separability surviving when magnitude is discarded"
    )


DISPLAY = {"speech": "Speech", "har": "HAR", "pendigits": "Pendigits"}


def plot_correlation_heatmaps(results: list[dict], out_path: Path):
    """Side-by-side averaged within-class correlation matrices, one per dataset.

    Near-diagonal (white off-diagonal) => a diagonal covariance loses little;
    visible off-diagonal colour => structure only a full covariance can model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, r in zip(axes, results):
        im = ax.imshow(r["_mean_corr"], vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_title(
            f"{DISPLAY.get(r['dataset'], r['dataset'])}\n"
            f"mean $|$off-diag$|$ = {r['mean_abs_offdiag_corr']:.2f}",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="within-class correlation")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(prog="python -m src.analysis.geometry")
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["speech", "har", "pendigits"],
        choices=["speech", "har", "pendigits"],
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="If set, save the within-class correlation heatmap PDF here.",
    )
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = []
    for ds in args.datasets:
        log.info("=== %s (split=%s) ===", ds, args.split)
        results.append(analyse_dataset(ds, args.split, device))

    print_table(results)

    if args.figures_dir is not None:
        plot_correlation_heatmaps(results, args.figures_dir / "within_class_correlation.pdf")

    # Per-class detail for the verdict / caveats.
    print("\nPer-class detail:")
    for r in results:
        print(f"\n  {r['dataset']} ({r['split']}, K={r['n_classes']}):")
        for pc in r["_per_class"]:
            print(
                f"    class {str(pc['label']):>8}: "
                f"cos={pc['cos_mean']:.3f}  lead_ev={pc['lead_eig_share']:.3f}  "
                f"effRank={pc['eff_rank']:.2f}  offdiag={pc['mean_abs_offdiag_corr']:.3f}"
            )


if __name__ == "__main__":
    main()
