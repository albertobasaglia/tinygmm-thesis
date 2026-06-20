"""
Export the measured-vs-estimated inference cost artifacts.

Parses the on-device benchmark log produced by firmware/bench (ESP32-S3,
microseconds per inference, min over trials) and pairs every measured
configuration with the analytical inference-FLOP count from the adapter cost
models in adapters.py. This is the hardware validation of the resource
accounting: the sweeps and all reported resource numbers use the analytical
counts; the benchmark exists to show that those counts track what the silicon
actually does.

Usage:
    python -m src.compare.export_bench
    python -m src.compare.export_bench --bench firmware/bench/output.txt \\
        --out tinygmm-tex/figures/resource --tables tinygmm-tex/tables

Writes:
    <out>/bench_us_vs_flops.pdf       (label: fig:bench_us_vs_flops; D=16, main text)
    <out>/bench_dim_us_vs_flops.pdf   (label: fig:bench_dim; all D, appendix)
    <tables>/bench.tex                (label: tab:bench; D=16, main text)
    <tables>/bench_full.tex           (label: tab:bench-full; all D, appendix)

The accuracy and personalization experiments use D=16 throughout, so the main
text reports only the D=16 benchmark. The full sweep additionally covers D=32 to
confirm that the analytical cost model tracks the measured time as the embedding
dimension changes; that cross-check is preserved in the appendix artifacts.
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import colors
from .adapters import (
    CosineAdapter,
    GMMAdapter,
    KNNAdapter,
    PrototypeAdapter,
    SmallAEAdapter,
)

ROOT = Path(__file__).parent.parent.parent

FULL_W = 5.12
H = 3.4

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

# One benchmark line, e.g.:
#   result small_ae D=16 L=4   us/inf= 1.38  cyc/inf= 54.3  sink=0.183032
#   result gmm D=16 K=1 cov=diag ...
#   result knn D=16 N=10 k=5 ...
_LINE = re.compile(
    r"result\s+(?P<name>\w+)\s+D=(?P<D>\d+)"
    r"(?:\s+L=(?P<L>\d+))?"
    r"(?:\s+K=(?P<K>\d+)\s+cov=(?P<cov>\w+))?"
    r"(?:\s+N=(?P<N>\d+)\s+k=(?P<k>\d+))?"
    r"\s+us/inf=\s*(?P<us>[\d.]+)"
)


def _inference_flops(cfg: dict) -> int:
    """Analytical per-sample inference FLOPs for one benchmarked config,
    taken from the same adapter cost models that produce every reported
    resource number. The synthetic fit only populates the shapes the cost
    model reads; the count does not depend on the data."""
    D = cfg["D"]
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg["name"] == "gmm":
            a = GMMAdapter(n_components=cfg["K"], covariance_type=cfg["cov"],
                           train_n=50, seed=0)
            a.fit(rng.standard_normal((50, D)).astype(np.float32))
        elif cfg["name"] == "knn":
            a = KNNAdapter(k=cfg["k"], train_n=cfg["N"])
            a.fit(rng.standard_normal((cfg["N"], D)).astype(np.float32))
        elif cfg["name"] == "small_ae":
            a = SmallAEAdapter(input_dim=D, latent_dim=cfg["L"], epochs=1,
                               train_n=20, seed=0)
            a.fit(rng.standard_normal((20, D)).astype(np.float32))
        elif cfg["name"] == "prototype":
            a = PrototypeAdapter(train_n=50)
            a.fit(rng.standard_normal((50, D)).astype(np.float32))
        elif cfg["name"] == "cosine":
            a = CosineAdapter(train_n=50)
            a.fit(rng.standard_normal((50, D)).astype(np.float32))
        else:
            raise ValueError(f"unknown benchmark adapter {cfg['name']!r}")
    return a.inference_flops()


def parse_bench(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        m = _LINE.search(line)
        if m is None:
            continue
        g = m.groupdict()
        cfg = {
            "name": g["name"],
            "D": int(g["D"]),
            "L": int(g["L"]) if g["L"] else None,
            "K": int(g["K"]) if g["K"] else None,
            "cov": g["cov"],
            "N": int(g["N"]) if g["N"] else None,
            "k": int(g["k"]) if g["k"] else None,
            "us": float(g["us"]),
        }
        cfg["flops"] = _inference_flops(cfg)
        rows.append(cfg)
    if not rows:
        raise SystemExit(f"no benchmark lines found in {path}")
    return rows


def _family(cfg: dict) -> str:
    if cfg["name"] == "gmm":
        return f"GMM {cfg['cov']}"
    if cfg["name"] == "knn":
        return f"kNN k={cfg['k']}"
    if cfg["name"] == "prototype":
        return "Prototype"
    if cfg["name"] == "cosine":
        return "Cosine"
    return "AE"


def _label(cfg: dict) -> str:
    if cfg["name"] == "gmm":
        return f"GMM K={cfg['K']} {cfg['cov']}"
    if cfg["name"] == "knn":
        return f"kNN N={cfg['N']} k={cfg['k']}"
    if cfg["name"] == "prototype":
        return "Prototype"
    if cfg["name"] == "cosine":
        return "Cosine"
    return f"AE L={cfg['L']}"


def fig_us_vs_flops(rows: list[dict], out_dir: Path, fname: str):
    families: dict[str, list[dict]] = {}
    for r in rows:
        families.setdefault(_family(r), []).append(r)

    dims = sorted({r["D"] for r in rows})
    fig, ax = plt.subplots()
    markers = {16: "o", 32: "s"}
    for fam, frows in families.items():
        frows = sorted(frows, key=lambda r: r["flops"])
        # Identity color for the family (shared with every other figure); the
        # family name already encodes covariance for GMM ("GMM diag"/"GMM full").
        color = colors.color_for_label(fam)
        for D in dims:
            drows = [r for r in frows if r["D"] == D]
            if not drows:
                continue
            ax.plot(
                [r["flops"] for r in drows], [r["us"] for r in drows],
                marker=markers[D], linestyle="--", linewidth=0.8,
                color=color, markerfacecolor="none" if D == 32 else None,
                label=fam if D == dims[0] else None,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Analytical inference FLOPs")
    ax.set_ylabel("Measured time per inference [µs]")
    ax.set_title("Measured on-device cost vs structural FLOP count")
    ax.grid(alpha=0.3, which="both")
    title = "filled: D=16, hollow: D=32" if len(dims) > 1 else "D=16"
    leg = ax.legend(title=title)
    leg.get_title().set_fontsize(8)
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def write_table(rows: list[dict], tables_dir: Path, fname: str,
                label: str, caption: str):
    tables_dir.mkdir(parents=True, exist_ok=True)
    body = []
    last_fam = None
    for r in sorted(rows, key=lambda r: (_family(r), r["D"], r["flops"])):
        fam = _family(r)
        if last_fam is not None and fam != last_fam:
            body.append("    \\midrule")
        last_fam = fam
        ns_per_flop = 1e3 * r["us"] / r["flops"]
        body.append(
            f"    {_label(r)} & {r['D']} & {r['flops']} & "
            f"{r['us']:.2f} & {ns_per_flop:.1f} \\\\"
        )
    tex = "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        # Units go in plain bracketed text with a literal "µ" rather than
        # siunitx (\si{\micro\second}): this table is shared between the DTU and
        # UniPD builds, and UniPD's DEIthesis.cls does not load siunitx, so
        # \si{...} would break that build. Both builds use xelatex, which reads
        # the source as UTF-8, so the literal "µ" renders correctly in both.
        "    Configuration & $D$ & Inference FLOPs & "
        "Time per inference [µs] & Time per FLOP [ns] \\\\",
        "    \\midrule",
        "\n".join(body),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    path = tables_dir / fname
    path.write_text(tex + "\n")
    print(f"  saved {path}")


def _print_stats(rows: list[dict], tag: str):
    """Log-domain fit quality and pairwise-ordering agreement for a row set.

    Reports the statistics the results chapter cites: the log-log correlation,
    the power-law slope, the worst miss of a single global constant times the
    count, and how many pairwise cost orderings invert against the silicon.
    """
    flops = np.array([r["flops"] for r in rows], float)
    us = np.array([r["us"] for r in rows], float)
    logf, logu = np.log(flops), np.log(us)
    r_log = float(np.corrcoef(logf, logu)[0, 1])
    slope = float(np.polyfit(logf, logu, 1)[0])
    logc = float(np.mean(logu - logf))            # best single global constant
    max_miss = float(np.exp(np.max(np.abs(logu - logf - logc))))

    n = len(rows)
    n_pairs = n * (n - 1) // 2
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            df, du = flops[i] - flops[j], us[i] - us[j]
            if df == 0:
                continue
            if (df > 0) != (du > 0):
                inv += 1
    print(f"\n  [{tag}] n={n}  log-log r={r_log:.3f}  slope={slope:.2f}  "
          f"max-miss-factor={max_miss:.2f}  inversions={inv}/{n_pairs}")


def main():
    parser = argparse.ArgumentParser(prog="python -m src.compare.export_bench")
    parser.add_argument("--bench", type=Path,
                        default=ROOT / "firmware" / "bench" / "output.txt",
                        help="Benchmark log from firmware/bench.")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "tinygmm-tex" / "figures" / "resource",
                        help="Directory for the comparison figure.")
    parser.add_argument("--tables", type=Path,
                        default=ROOT / "tinygmm-tex" / "tables",
                        help="Directory for bench.tex.")
    args = parser.parse_args()

    rows = parse_bench(args.bench)
    rows16 = [r for r in rows if r["D"] == 16]
    args.out.mkdir(parents=True, exist_ok=True)

    fmt = "  {:<22} {:>4} {:>7} {:>9} {:>9}"
    print(fmt.format("Config", "D", "FLOPs", "us/inf", "ns/FLOP"))
    print(fmt.format("------", "-", "-----", "------", "-------"))
    for r in sorted(rows, key=lambda r: (_family(r), r["D"], r["flops"])):
        print(fmt.format(_label(r), r["D"], r["flops"], f"{r['us']:.2f}",
                         f"{1e3 * r['us'] / r['flops']:.1f}"))

    _print_stats(rows16, "D=16 (main text)")
    _print_stats(rows, "all D (appendix)")

    main_caption = (
        "Measured per-inference time on the ESP32-S3 benchmark against the"
        " analytical inference-FLOP count of each configuration, at the"
        " embedding dimension $D=16$ used throughout the experiments. The last"
        " column is the implied time per counted FLOP; it is stable within each"
        " computational pattern and varies across patterns because transcendental"
        " operations are charged as one FLOP each."
    )
    full_caption = (
        "Full benchmark sweep, extending Table~\\ref{tab:bench} with $D=32$"
        " configurations. The accuracy experiments use $D=16$ only; the $D=32$"
        " measurements are included solely to check that the analytical cost"
        " model tracks the measured time as the embedding dimension changes,"
        " confirming the quadratic-in-$D$ growth of the full covariance and the"
        " linear-in-$D$ growth of the diagonal and distance kernels on silicon."
    )

    fig_us_vs_flops(rows16, args.out, "bench_us_vs_flops.pdf")
    fig_us_vs_flops(rows, args.out, "bench_dim_us_vs_flops.pdf")
    write_table(rows16, args.tables, "bench.tex", "tab:bench", main_caption)
    write_table(rows, args.tables, "bench_full.tex", "tab:bench-full",
                full_caption)


if __name__ == "__main__":
    main()
