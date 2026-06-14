"""
Export the dataset-independent resource artifacts.

Per-adapter inference FLOPs and stored-parameter counts are
deterministic structural counts: they depend only on the adapter and the
embedding dimension D, not on the dataset or on any sweep result. All three
thesis datasets (speech, HAR, Pendigits) share D=16 and the same selected
adapter configurations, so a single table and a single pair of FLOPs figures
apply throughout. This script generates them directly from the adapter cost
models, so they never need a sweep to be run and can never go stale relative
to the FLOP accounting in adapters.py.

Usage:
    python -m src.compare.export_resource
    python -m src.compare.export_resource --out tinygmm-tex/figures/resource \\
        --tables tinygmm-tex/tables --dim 16

Writes:
    <out>/inference_flops_bar.pdf
    <out>/enrollment_flops_bar.pdf
    <out>/inference_flops_vs_train_n.pdf
    <tables>/resource.tex            (label: tab:resource)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .adapters import (
    GMMAdapter,
    KNNAdapter,
    SmallAEAdapter,
    CosineAdapter,
    PrototypeAdapter,
)

ROOT = Path(__file__).parent.parent.parent

FULL_W = 5.12
H = 3.0

plt.rcParams.update({
    "figure.figsize": (FULL_W, H),
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

FIXED_TRAIN_N = 50


def _adapter_specs(D: int):
    """(label, factory) per adapter; factory(n) builds it for enrollment size n.

    Labels and configurations mirror BEST_LINES in export_plots.py so the
    resource artifacts line up with the rest of the results chapter.
    """
    return [
        ("GMM K=1 diag", lambda n: GMMAdapter(n_components=1, covariance_type="diag", train_n=n, seed=0)),
        ("GMM K=1 full", lambda n: GMMAdapter(n_components=1, covariance_type="full", train_n=n, seed=0)),
        ("kNN k=5",      lambda n: KNNAdapter(k=5, train_n=n)),
        # epochs=100 matches the selected AE config in configs/frozen.py, so the
        # enrollment-FLOPs column reflects the real fit cost (inference FLOPs and
        # parameter count do not depend on epochs).
        ("SmallAE",      lambda n: SmallAEAdapter(input_dim=D, latent_dim=8, epochs=100, train_n=n, seed=0)),
        ("Cosine",       lambda n: CosineAdapter(train_n=n)),
        ("Prototype",    lambda n: PrototypeAdapter(train_n=n)),
    ]


def _costs(D: int, n: int) -> dict[str, tuple[int, int, int]]:
    """Structural (inference FLOPs, stored parameters, one-time enrollment FLOPs)
    per adapter at enrollment size n. The adapters are fit on synthetic data only
    to populate the shapes their cost models read; the counts do not depend on the
    data (the GMM's EM converges in the same 2 iterations across all three real
    datasets, so its enrollment cost is dataset-independent too)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, D)).astype(np.float32)
    out: dict[str, tuple[int, int, int]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ill-conditioned cov / convergence at tiny n
        for label, factory in _adapter_specs(D):
            adapter = factory(n)
            adapter.fit(X)
            # Report the best-case GMM enrollment: a single closed-form pass
            # (one weighted mean and covariance). For K=1 the MLE is exact in one
            # pass, whereas sklearn's EM wrapper runs ~2 iterations; charging one
            # iteration is the lowest achievable fit cost.
            gmm = getattr(adapter, "_gmm", None)
            if gmm is not None:
                gmm.n_iter_ = 1
            out[label] = (adapter.inference_flops(), adapter.parameters(),
                          adapter.training_flops())
    return out


def _grp(n: int) -> str:
    """Integer with thin-space thousands separators for a tabular cell, e.g.
    11256000 -> '11\\,256\\,000'."""
    return f"{n:,}".replace(",", "\\,")


def write_table(D: int, train_n: int, tables_dir: Path):
    tables_dir.mkdir(parents=True, exist_ok=True)
    costs = _costs(D, train_n)
    rows = []
    for label, _ in _adapter_specs(D):
        infer, params, enroll = costs[label]
        rows.append(f"    {label} & {_grp(infer)} & {_grp(enroll)} & {_grp(params)} \\\\")
    tex = "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Resource cost per adaptive layer at \\texttt{{train\\_n}}={train_n}:"
        " per-sample inference FLOPs, one-time enrollment (fit) FLOPs, and stored"
        " parameters. These are deterministic structural counts that depend"
        f" only on the adaptive layer and the embedding dimension ($D={D}$, shared by all"
        " three datasets), not on the dataset, so a single table applies"
        " throughout and no confidence interval is reported. The enrollment count"
        " is the cost of the one-time fit (the autoencoder's 100-epoch training"
        " loop, a single mean for cosine and the prototype); for the GMM it is the"
        " best case of a single closed-form pass over the enrollment set, one"
        " weighted mean and covariance, since for one component the maximum-likelihood"
        " fit is exact in one pass. The k-nearest-neighbor baseline only stores its"
        " vectors and so does no fitting arithmetic.}",
        "  \\label{tab:resource}",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    Adaptive layer & Inference FLOPs & Enrollment FLOPs & Parameters \\\\",
        "    \\midrule",
        "\n".join(rows),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    path = tables_dir / "resource.tex"
    path.write_text(tex + "\n")
    print(f"  saved {path}")


def fig_bar(D: int, train_n: int, out_dir: Path):
    costs = _costs(D, train_n)
    labels = [label for label, _ in _adapter_specs(D)]
    flops = [costs[label][0] for label in labels]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, flops)
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("Inference FLOPs")
    ax.set_title(f"Inference cost per sample (train_n={train_n})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "inference_flops_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_enroll_bar(D: int, train_n: int, out_dir: Path):
    """One-time enrollment (fit) FLOPs per adapter, log y-axis.

    Mirrors fig_bar but for the enrollment cost, which spans several orders of
    magnitude (from the prototype's single mean to the autoencoder's training
    loop), so a log scale is used. Adapters with zero fit cost (kNN only stores
    its vectors) are drawn at the axis floor and labelled "0"."""
    costs = _costs(D, train_n)
    labels = [label for label, _ in _adapter_specs(D)]
    enroll = [costs[label][2] for label in labels]

    floor = 0.5  # log axis has no zero; draw zero-cost bars as an invisible stub
    heights = [v if v > 0 else floor for v in enroll]
    bar_labels = [f"{v:,}" for v in enroll]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, heights)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.bar_label(bars, labels=bar_labels)
    ax.set_ylabel("Enrollment FLOPs (one-time)")
    ax.set_title(f"Enrollment cost (train_n={train_n})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "enrollment_flops_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_vs_train_n(D: int, ns: list[int], out_dir: Path):
    labels = [label for label, _ in _adapter_specs(D)]
    series = {label: [] for label in labels}
    for n in ns:
        costs = _costs(D, n)
        for label in labels:
            series[label].append(costs[label][0])

    fig, ax = plt.subplots()
    for label in labels:
        ax.plot(ns, series[label], label=label)
    ax.set_xlabel("Enrollment size (train_n)")
    ax.set_ylabel("Inference FLOPs per sample")
    ax.set_title("Inference cost vs enrollment size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "inference_flops_vs_train_n.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _parse_grid(spec: str) -> list[int]:
    start, stop, step = (int(x) for x in spec.split(":"))
    return list(range(start, stop, step))


def main():
    parser = argparse.ArgumentParser(prog="python -m src.compare.export_resource")
    parser.add_argument("--out", type=Path, default=ROOT / "tinygmm-tex" / "figures" / "resource",
                        help="Directory for the FLOPs figures.")
    parser.add_argument("--tables", type=Path, default=ROOT / "tinygmm-tex" / "tables",
                        help="Directory for resource.tex.")
    parser.add_argument("--dim", type=int, default=16,
                        help="Embedding dimension D (shared across datasets).")
    parser.add_argument("--train-n", type=int, default=FIXED_TRAIN_N,
                        help="Representative enrollment budget for the table and bar chart.")
    parser.add_argument("--n-grid", type=str, default="5:200:5",
                        help="start:stop:step enrollment grid for the vs-train_n figure.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ns = _parse_grid(args.n_grid)

    print(f"Resource artifacts for D={args.dim}, train_n={args.train_n}")
    print(f"Writing figures to {args.out}")
    print(f"Writing table to {args.tables}\n")

    costs = _costs(args.dim, args.train_n)
    fmt = "  {:<14} {:>10} {:>14} {:>12}"
    print(fmt.format("Adapter", "InfFLOPs", "EnrollFLOPs", "Params"))
    print(fmt.format("-------", "--------", "-----------", "-----"))
    for label, _ in _adapter_specs(args.dim):
        infer, params, enroll = costs[label]
        print(fmt.format(label, infer, enroll, params))
    print()

    write_table(args.dim, args.train_n, args.tables)
    fig_bar(args.dim, args.train_n, args.out)
    fig_enroll_bar(args.dim, args.train_n, args.out)
    fig_vs_train_n(args.dim, ns, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
