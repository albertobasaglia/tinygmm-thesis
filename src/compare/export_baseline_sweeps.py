"""Appendix tables for the baseline hyperparameter sweeps.

The GMM ablation is presented in full in the experiments chapter, but the
autoencoder and k-NN baselines are only tuned-then-fixed: the chapters report
the selected configuration, not the sweep behind it. These appendix tables
surface that sweep so the selection is reproducible.

Both baselines are selected on the *validation* classes (the classes in the
``sweep_<dataset>_baseline_latest`` parquets, disjoint from the held-out test).
Cells are validation ACC@FAR=5% (mean $\\pm$ 95% CI across target-class x trial
groups) at the fixed enrollment budget. Columns are the three datasets; the
selected configuration is the bold row.

Run from the repo root::

    python -m src.compare.export_baseline_sweeps

Writes ``tinygmm-tex/tables/ae_sweep.tex`` and ``.../knn_sweep.tex``.
"""

from pathlib import Path

import pandas as pd

from src.compare.export_plots import (
    FIXED_TRAIN_N,
    ROOT,
    _cell,
    _pretty_dataset,
)

RESULTS = ROOT / "results"
TABLES = ROOT / "tinygmm-tex" / "tables"
DATASETS = ["speech", "har", "pendigits"]


def _load(dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / f"sweep_{dataset}_baseline_latest.parquet")
    return df[df["p_train_n"] == FIXED_TRAIN_N]


def _table(caption: str, label: str, header: str, rows: list[str]) -> str:
    ncols = len(DATASETS)
    return "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{l{'r' * ncols}}}",
        "    \\toprule",
        f"    {header} & " + " & ".join(_pretty_dataset(d) for d in DATASETS) + " \\\\",
        "    \\midrule",
        "\n".join(rows),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])


def _row(label: str, slices: list[pd.Series], bold: bool) -> str:
    cells = ["--" if s.empty else _cell(s, bold=bold) for s in slices]
    label = f"{{\\boldmath\\bfseries {label}}}" if bold else label
    return f"    {label} & " + " & ".join(cells) + " \\\\"


def _ae_table(frames: dict[str, pd.DataFrame]) -> str:
    configs = [(4, 50), (4, 100), (8, 50), (8, 100)]  # selected: (8, 100)
    rows = []
    for L, E in configs:
        slices = []
        for d in DATASETS:
            ae = frames[d][frames[d]["p_adapter"] == "SmallAEAdapter"]
            slices.append(ae[(ae["p_latent_dim"] == L) & (ae["p_epochs"] == E)]["m_acc_at_far5"])
        rows.append(_row(f"$L={L}$, {E} epochs", slices, bold=(L, E) == (8, 100)))
    caption = (
        "Autoencoder baseline selection: validation ACC@FAR=5\\% (mean $\\pm$ 95\\%"
        f" CI across target classes $\\times$ trials) at \\texttt{{train\\_n}}={FIXED_TRAIN_N},"
        " over the latent dimension $L$ and training epochs. Higher is better. The"
        " bold row, $L=8$ with 100 epochs, is the strongest on every dataset and is"
        " the configuration used in the held-out test.")
    return _table(caption, "tab:ae_sweep", "Configuration", rows)


def _knn_table(frames: dict[str, pd.DataFrame]) -> str:
    ks = sorted({int(k) for d in DATASETS
                 for k in frames[d][frames[d]["p_adapter"] == "KNNAdapter"]["p_k"].dropna().unique()})
    rows = []
    for k in ks:
        slices = []
        for d in DATASETS:
            knn = frames[d][frames[d]["p_adapter"] == "KNNAdapter"]
            slices.append(knn[knn["p_k"] == k]["m_acc_at_far5"])
        rows.append(_row(f"$k={k}$", slices, bold=(k == 1)))
    caption = (
        "k-nearest-neighbor baseline selection: validation ACC@FAR=5\\% (mean $\\pm$"
        f" 95\\% CI across target classes $\\times$ trials) at \\texttt{{train\\_n}}={FIXED_TRAIN_N},"
        " over the neighbor count $k$. Higher is better. The bold row, $k=1$, is the"
        " configuration used in the held-out test.")
    return _table(caption, "tab:knn_sweep", "Neighbors", rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    frames = {d: _load(d) for d in DATASETS}
    for name, tex in [("ae_sweep", _ae_table(frames)),
                      ("knn_sweep", _knn_table(frames))]:
        path = TABLES / f"{name}.tex"
        path.write_text(tex + "\n")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
