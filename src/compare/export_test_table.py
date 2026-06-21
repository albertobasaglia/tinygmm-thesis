"""
Export a combined held-out-test summary table across all datasets.

One row per adapter (in BEST_LINES order), one column per dataset; each cell is
ACC@FAR=5% mean +/- 95% CI at train_n=50, and the best adapter on each dataset is
shown in bold. This is the cross-dataset "scoreboard" that sits above the
per-dataset blocks in the results chapter; the per-dataset tables keep EER and the
side-by-side CI chart.

Usage:
    python -m src.compare.export_test_table \\
        results/test_speech.parquet results/test_har.parquet results/test_pendigits.parquet \\
        --out tinygmm-tex/tables/test_summary_all.tex
"""

import argparse
from pathlib import Path

import pandas as pd

from .export_plots import BEST_LINES, FIXED_TRAIN_N, _ci95, _filter, _pretty_dataset


def _cell(mean: float, ci: float, bold: bool) -> str:
    # \boldmath (not \textbf) is what actually bolds math-mode content.
    cell = f"${mean:.3f} \\pm {ci:.3f}$"
    return f"{{\\boldmath {cell}}}" if bold else cell


def main():
    parser = argparse.ArgumentParser(prog="python -m src.compare.export_test_table")
    parser.add_argument("parquets", type=Path, nargs="+",
                        help="held-out test parquets, e.g. results/test_speech.parquet ...")
    parser.add_argument("--out", type=Path, required=True,
                        help="output .tex path, e.g. tinygmm-tex/tables/test_summary_all.tex")
    args = parser.parse_args()

    # Per dataset: {adapter label -> (mean, ci)} for ACC@FAR=5% at train_n=50.
    datasets = []  # (pretty_name, {label: (mean, ci)})
    for pq in args.parquets:
        key = pq.stem.removeprefix("test_")
        sub = pd.read_parquet(pq)
        sub = sub[sub["p_train_n"] == FIXED_TRAIN_N]
        vals = {}
        for label, where in BEST_LINES:
            s = _filter(sub, where)
            if not s.empty:
                vals[label] = _ci95(s["m_acc_at_far5"])
        datasets.append((_pretty_dataset(key), vals))

    # Rows: BEST_LINES labels present in any dataset, in BEST_LINES order.
    labels = [label for label, _ in BEST_LINES
              if any(label in vals for _, vals in datasets)]
    # Best adapter per dataset = highest mean ACC@FAR=5%.
    best = {pretty: (max(vals, key=lambda l: vals[l][0]) if vals else None)
            for pretty, vals in datasets}

    rows = []
    for label in labels:
        cells = []
        for pretty, vals in datasets:
            if label in vals:
                mean, ci = vals[label]
                cells.append(_cell(mean, ci, best[pretty] == label))
            else:
                cells.append("--")
        rows.append(f"    {label} & " + " & ".join(cells) + " \\\\")

    ncol = len(datasets)
    tex = "\n".join([
        "\\begin{table}[tbp]",
        "  \\centering",
        "  \\caption{Held-out test \\texttt{ACC@FAR=5\\%} at \\texttt{train\\_n}="
        f"{FIXED_TRAIN_N} (mean $\\pm$ 95\\% CI), per dataset; higher is better. The"
        " best adapter on each dataset is in \\textbf{bold}.}",
        "  \\label{tab:test_summary_all}",
        f"  \\begin{{tabular}}{{l{'r' * ncol}}}",
        "    \\toprule",
        "    Adapter & " + " & ".join(pretty for pretty, _ in datasets) + " \\\\",
        "    \\midrule",
        "\n".join(rows),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
