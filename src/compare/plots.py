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


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask for Pareto-optimal points (lower x, lower y preferred)."""
    is_pareto = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        # A point is dominated if another point has <= x and <= y, with at least one <
        dominated = ((x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i])))
        dominated[i] = False  # don't compare with self
        if dominated.any():
            is_pareto[i] = False
    return is_pareto


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
        labels.append(label)
        precisions.append(subset["m_precision"].mean())
        recalls.append(subset["m_recall"].mean())

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


def plot_loss_curves(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """Training loss at 5 evenly-spaced checkpoints for AE adapters.

    Shows whether the AE has converged or still needs more epochs.
    X-axis is the fraction of training completed (0.2, 0.4, ..., 1.0).
    Each line averages over all matching rows (target words, trials, train_n).
    Filter via the ``where`` dicts to narrow the view.
    """
    fig, ax = plt.subplots()
    fracs = np.linspace(0.2, 1.0, 5)
    loss_cols = [f"m_loss_{i}" for i in range(1, 6)]

    for label, where in lines:
        subset = _filter(df, where).dropna(subset=loss_cols, how="all")
        if subset.empty:
            continue
        means = subset[loss_cols].mean().values
        stds = subset[loss_cols].std().values
        line, = ax.plot(fracs, means, marker="o", label=label)
        if not np.all(np.isnan(stds)):
            ax.fill_between(fracs, means - stds, means + stds,
                            alpha=0.15, color=line.get_color())

    ax.set_xlabel("Fraction of training completed")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title("AE training convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_loss_vs_eer(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """Scatter of final training loss vs EER.

    Shows whether lower reconstruction loss actually translates to better
    anomaly detection. If there is no correlation, the reconstruction
    objective is misaligned with the task.
    """
    fig, ax = plt.subplots()
    for label, where in lines:
        subset = _filter(df, where).dropna(subset=["m_loss_5"])
        if subset.empty:
            continue
        ax.scatter(subset["m_loss_5"], subset["m_eer"], alpha=0.5, s=30, label=label)

    ax.set_xlabel("Final training loss (MSE)")
    ax.set_ylabel("EER (lower is better)")
    ax.set_title("Does lower reconstruction loss → better anomaly detection?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_gmm_components(df: pd.DataFrame, y: str = "m_eer",
                        fixed_train_n: int = 45):
    """Bar chart: metric vs n_components, grouped by covariance type.

    Shows the optimal number of components for each covariance structure.
    Only includes GMMAdapter rows.

    Args:
        df            : results DataFrame
        y             : metric column (e.g. "m_eer", "m_auc")
        fixed_train_n : training budget to slice on
    """
    sub = _filter(df, {"p_adapter": "GMMAdapter"})
    sub = sub[sub["p_train_n"] == fixed_train_n]

    fig, ax = plt.subplots()
    cov_types = sorted(sub["p_covariance_type"].unique())
    components = sorted(sub["p_n_components"].unique())
    x = np.arange(len(components))
    width = 0.8 / max(len(cov_types), 1)

    for i, cov in enumerate(cov_types):
        means = []
        stds = []
        for k in components:
            vals = sub[(sub["p_n_components"] == k) & (sub["p_covariance_type"] == cov)][y]
            means.append(vals.mean())
            stds.append(vals.std())
        offset = (i - len(cov_types) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=cov)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in components])
    ax.set_ylabel(y.replace("m_", "").upper())
    ax.set_title(f"GMM: {y.replace('m_', '')} vs n_components (train_n={fixed_train_n})")
    ax.legend(title="covariance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def plot_gmm_diag_vs_full(df: pd.DataFrame, y: str = "m_eer"):
    """Line plot: metric vs train_n, one line per (K, covariance_type) pair.

    Shows whether full covariance helps and how it interacts with K.

    Args:
        df : results DataFrame
        y  : metric column (e.g. "m_eer", "m_auc")
    """
    sub = _filter(df, {"p_adapter": "GMMAdapter"})
    fig, ax = plt.subplots()

    for cov in sorted(sub["p_covariance_type"].unique()):
        for k in sorted(sub["p_n_components"].unique()):
            s = sub[(sub["p_n_components"] == k) & (sub["p_covariance_type"] == cov)]
            label = f"K={k} {cov}"
            _plot_line(ax, s, "p_train_n", y, label)

    ax.set_xlabel("train_n")
    ax.set_ylabel(y.replace("m_", "").upper())
    ax.set_title(f"GMM: {y.replace('m_', '')} — diag vs full covariance")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()




def plot_pareto(df: pd.DataFrame, lines: list[tuple[str, dict]],
                x: str = "m_training_macs", y: str = "m_eer"):
    """Scatter plot with Pareto-optimal points highlighted per method.

    Shows which methods dominate at different MAC budgets. Pareto-optimal
    points (no other point has both lower MACs AND lower EER) are highlighted
    with larger markers and connected by lines.

    Args:
        df    : results DataFrame
        lines : list of (label, filter_dict) pairs
        x     : column for x-axis (default: m_training_macs)
        y     : column for y-axis (default: m_eer, lower is better)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, where in lines:
        subset = _filter(df, where)
        if subset.empty:
            continue

        # Aggregate by unique (x, y) combinations across trials/configs
        agg = subset.groupby([x])[y].mean().reset_index()
        xs, ys = agg[x].values, agg[y].values

        # All points as small markers
        ax.scatter(xs, ys, alpha=0.4, s=20, label=None)

        # Pareto frontier: larger markers + line
        pareto = _pareto_mask(xs, ys)
        if pareto.any():
            px, py = xs[pareto], ys[pareto]
            order = np.argsort(px)
            color = ax.scatter(px, py, s=80, label=label, zorder=3).get_facecolors()[0]
            ax.plot(px[order], py[order], color=color, linewidth=1.5, alpha=0.7, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Training MACs")
    ax.set_ylabel("EER (lower is better)")
    ax.set_title("Pareto Frontier: EER vs Training MACs")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
