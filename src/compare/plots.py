import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from . import colors


def _filter(df: pd.DataFrame, where: dict) -> pd.DataFrame:
    subset = df.copy()
    for k, v in where.items():
        if k not in subset.columns:
            return subset.iloc[0:0]
        subset = subset[subset[k] == v]
    return subset


def _agg(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return df.groupby(x)[y].agg(["mean", "std", "count"]).reset_index().sort_values(x)


def _plot_line(ax, subset: pd.DataFrame, x: str, y: str, label: str, **kwargs):
    agg = _agg(subset, x, y)
    line, = ax.plot(agg[x], agg["mean"], label=label, **kwargs)
    if agg["std"].notna().any():
        sem = agg["std"] / np.sqrt(agg["count"])
        t_crit = stats.t.ppf(0.975, df=agg["count"] - 1)
        margin = t_crit * sem
        ax.fill_between(agg[x], agg["mean"] - margin, agg["mean"] + margin,
                        alpha=0.15, color=line.get_color())


def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lines(df: pd.DataFrame, x: str, y: str,
               lines: list[tuple[str, dict]], out_path,
               title: str | None = None, ylabel: str | None = None,
               line_colors: list | None = None):
    """One line per (label, filter_dict), mean + 95% CI band. Saves PDF.

    Colors come from the central scheme (colors.color_for) so an algorithm keeps
    its color across figures. Pass line_colors (one per line, same order) only
    for within-figure ablations whose series are not the comparison families.
    """
    fig, ax = plt.subplots()
    for i, (label, where) in enumerate(lines):
        color = line_colors[i] if line_colors is not None else colors.color_for(where)
        _plot_line(ax, _filter(df, where), x, y, label, color=color)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} vs {x}")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_ci_bars(df: pd.DataFrame, lines: list[tuple[str, dict]],
                 train_n: int, y: str, out_path,
                 title: str | None = None, xlabel: str | None = None,
                 line_colors: list | None = None):
    """Horizontal error-bar chart: one row per (label, filter_dict), mean +/- 95% CI.

    Colors come from the central scheme (colors.color_for); pass line_colors only
    for within-figure ablations (see plot_lines).
    """
    sub = df[df["p_train_n"] == train_n]
    fig, ax = plt.subplots()
    for i, (label, where) in enumerate(lines):
        vals = _filter(sub, where)[y].dropna()
        if vals.empty:
            continue
        n = len(vals)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci = t_crit * sem
        color = line_colors[i] if line_colors is not None else colors.color_for(where)
        ax.errorbar(mean, i, xerr=ci, fmt="o", capsize=5, markersize=6, color=color)
        ax.text(mean + ci + 0.005, i, f"{mean:.3f}", va="center", fontsize=9)

    ax.set_yticks(range(len(lines)))
    ax.set_yticklabels([c[0] for c in lines])
    ax.set_xlabel(xlabel or y)
    ax.set_title(title or f"{y} with 95% CI (train_n={train_n})")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_path)


def plot_gmm_grid(df: pd.DataFrame, train_n: int, out_path,
                  y: str = "m_eer"):
    """Grouped bar chart: n_components on x-axis, one bar per covariance type."""
    sub = _filter(df, {"p_adapter": "GMMAdapter"})
    sub = sub[sub["p_train_n"] == train_n]
    if sub.empty:
        print(f"plot_gmm_grid: no GMM rows at train_n={train_n}, skipping.")
        return

    cov_types = sorted(sub["p_covariance_type"].unique())
    components = sorted(sub["p_n_components"].unique())
    x = np.arange(len(components))
    width = 0.8 / max(len(cov_types), 1)

    fig, ax = plt.subplots()
    for i, cov in enumerate(cov_types):
        means, stds = [], []
        for k in components:
            vals = sub[(sub["p_n_components"] == k) & (sub["p_covariance_type"] == cov)][y]
            means.append(vals.mean())
            stds.append(vals.std())
        offset = (i - len(cov_types) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=cov,
               color=colors.COV.get(cov, colors.FALLBACK))

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in components])
    ax.set_ylabel(y.replace("m_", "").upper())
    ax.set_title(f"GMM: {y.replace('m_', '')} vs n_components (train_n={train_n})")
    ax.legend(title="covariance")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)
