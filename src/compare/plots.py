import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _filter(df: pd.DataFrame, where: dict) -> pd.DataFrame:
    subset = df.copy()
    for k, v in where.items():
        subset = subset[subset[k] == v]
    return subset


def _agg(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Group by x, compute mean and std of y (handles single-trial data)."""
    return df.groupby(x)[y].agg(["mean", "std"]).reset_index().sort_values(x)


def _plot_line(ax, subset: pd.DataFrame, x: str, y: str, label: str, **kwargs):
    """Plot mean line with ±1 std shaded band."""
    agg = _agg(subset, x, y)
    line, = ax.plot(agg[x], agg["mean"], marker="o", label=label, **kwargs)
    if agg["std"].notna().any():
        ax.fill_between(agg[x], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                         alpha=0.15, color=line.get_color())


def plot_far_recall(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """Scatter plot of operating points in FAR vs Recall space.

    Each (label, filter_dict) pair becomes one scatter series, with one point
    per row that matches the filter.  Useful for exposing degenerate configs
    (FAR≈1, recall≈1) versus well-calibrated ones.

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        subset = df.copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        ax.scatter(subset["m_false_alarm_rate"], subset["m_recall"], label=label, s=60)

    ax.set_xlabel("False Alarm Rate (FAR)")
    ax.set_ylabel("Recall (TPR)")
    ax.set_title("Operating points: Recall vs FAR")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_eer(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """EER vs train_n for selected configs.

    EER is threshold-free and directly comparable across adapters — lower is
    better.  A single line per config keeps the chart readable.

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        _plot_line(ax, _filter(df, where), "p_train_n", "m_eer", label)

    ax.set_xlabel("train_n")
    ax.set_ylabel("EER")
    ax.set_title("Equal Error Rate vs training budget")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()



def plot_precision_recall_bar(df: pd.DataFrame, train_n: int,
                               lines: list[tuple[str, dict]]):
    """Grouped bar chart of precision and recall at a fixed train_n.

    Shows threshold calibration quality: a well-calibrated adapter has both
    high recall and high precision.  Degenerate configs (recall≈1, precision≈0.5)
    are immediately visible.

    Args:
        df      : results DataFrame
        train_n : the training budget to slice on
        lines   : list of (label, filter_dict) pairs
    """
    labels, precisions, recalls = [], [], []
    for label, where in lines:
        subset = df[df["p_train_n"] == train_n].copy()
        for k, v in where.items():
            subset = subset[subset[k] == v]
        if subset.empty:
            continue
        row = subset.iloc[0]
        labels.append(label)
        precisions.append(row["m_precision"])
        recalls.append(row["m_recall"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, precisions, width, label="Precision")
    ax.bar(x + width / 2, recalls,    width, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision vs Recall at train_n={train_n}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def plot_f1(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """F1 vs train_n for selected configs.

    F1 penalises degenerate configs (recall=1, precision≈0.5 → F1≈0.66)
    while rewarding well-calibrated ones, making it a clean single-line
    comparison when the threshold matters.

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        _plot_line(ax, _filter(df, where), "p_train_n", "m_f1", label)

    ax.set_xlabel("train_n")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs training budget")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_eer_by_dim(df: pd.DataFrame, lines: list[tuple[str, dict]],
                    fixed_train_n: int = None):
    """EER vs embedding_dim, one line per adapter config.

    Use this to visualise the embedding-size ablation: how much can the
    representation be compressed before accuracy degrades?

    Args:
        df            : results DataFrame (must have an embedding_dim column)
        lines         : list of (label, filter_dict) pairs
        fixed_train_n : if set, filter to a single train_n before plotting
    """
    subset = df.copy()
    if fixed_train_n is not None:
        subset = subset[subset["p_train_n"] == fixed_train_n]

    fig, ax = plt.subplots()
    for label, where in lines:
        s = subset.copy()
        for k, v in where.items():
            s = s[s[k] == v]
        s = s.groupby("p_embedding_dim")["m_eer"].mean().reset_index().sort_values("p_embedding_dim")
        ax.plot(s["p_embedding_dim"], s["m_eer"], marker="o", label=label)

    ax.set_xlabel("embedding_dim")
    ax.set_ylabel("EER")
    title = "EER vs embedding dimension"
    if fixed_train_n is not None:
        title += f" (train_n={fixed_train_n})"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_eer_train_n_by_dim(df: pd.DataFrame, where: dict = None):
    """EER vs train_n with one line per embedding_dim.

    Shows whether a smaller embedding can be compensated by more enrollment
    data — a key question for TinyML deployment trade-offs.

    Args:
        df    : results DataFrame
        where : optional filter to narrow to a single adapter config,
                e.g. {"p_adapter": "GMMAdapter", "p_covariance_type": "diag"}
    """
    subset = df.copy()
    if where:
        for k, v in where.items():
            subset = subset[subset[k] == v]

    fig, ax = plt.subplots()
    for dim, group in subset.groupby("p_embedding_dim"):
        _plot_line(ax, group, "p_train_n", "m_eer", f"dim={dim}")

    ax.set_xlabel("train_n")
    ax.set_ylabel("EER")
    title = "EER vs train_n by embedding dimension"
    if where:
        title += f"\n[{', '.join(f'{k}={v}' for k, v in where.items())}]"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_sweep(df: pd.DataFrame, x: str, y: str, group_by: str = None,
               filter: str = None, where: dict = None):
    """Plot sweep results with x and y as column names, grouped into lines.

    Args:
        df       : results DataFrame
        x        : column name for the x-axis  (e.g. "p_n_components")
        y        : column name for the y-axis   (e.g. "m_auc", "m_recall")
        group_by : column name for separate lines (e.g. "p_covariance_type")
        filter   : if set, only plot rows where p_adapter == this name
        where    : extra column filters, e.g. {"p_covariance_type": "diag"}
    """
    subset = df.copy()
    if filter:
        subset = subset[subset["p_adapter"] == filter]
    if where:
        for k, v in where.items():
            subset = subset[subset[k] == v]

    fig, ax = plt.subplots()
    if group_by:
        for label, group in subset.groupby(group_by):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], marker="o", label=label)
    else:
        subset = subset.sort_values(x)
        ax.plot(subset[x], subset[y], marker="o")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    title = f"{filter or 'all'}: {y} vs {x}"
    if group_by:
        title += f" (grouped by {group_by})"
    if where:
        title += f" [{', '.join(f'{k}={v}' for k, v in where.items())}]"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_lines(df: pd.DataFrame, x: str, y: str,
               lines: list[tuple[str, dict]]):
    """Plot specific configs as named lines on one figure.

    Args:
        df    : results DataFrame
        x     : column for x-axis (e.g. "p_train_n")
        y     : column for y-axis (e.g. "m_auc")
        lines : list of (label, filter_dict) pairs — each becomes one line
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        _plot_line(ax, _filter(df, where), x, y, label)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
