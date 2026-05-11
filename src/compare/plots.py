import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def _filter(df: pd.DataFrame, where: dict) -> pd.DataFrame:
    subset = df.copy()
    for k, v in where.items():
        if k not in subset.columns:
            # Column absent (e.g. a sweep dimension that wasn't run): no rows match.
            return subset.iloc[0:0]
        subset = subset[subset[k] == v]
    return subset


def _agg(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Group by x, compute mean, std, and count of y."""
    return df.groupby(x)[y].agg(["mean", "std", "count"]).reset_index().sort_values(x)


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
    """Plot mean line with 95% confidence interval shaded band."""
    agg = _agg(subset, x, y)
    line, = ax.plot(agg[x], agg["mean"], marker="o", label=label, **kwargs)
    if agg["std"].notna().any():
        sem = agg["std"] / np.sqrt(agg["count"])
        t_crit = stats.t.ppf(0.975, df=agg["count"] - 1)
        margin = t_crit * sem
        ax.fill_between(agg[x], agg["mean"] - margin, agg["mean"] + margin,
                         alpha=0.15, color=line.get_color())


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
        s = _filter(subset, where)
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
    subset = _filter(df, where) if where else df.copy()

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
        y        : column name for the y-axis   (e.g. "m_auc", "m_eer")
        group_by : column name for separate lines (e.g. "p_covariance_type")
        filter   : if set, only plot rows where p_adapter == this name
        where    : extra column filters, e.g. {"p_covariance_type": "diag"}
    """
    subset = df.copy()
    if filter:
        subset = subset[subset["p_adapter"] == filter]
    if where:
        subset = _filter(subset, where)

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
    X-axis is epoch number (derived from p_epochs * fraction checkpoints).
    Each line averages over all matching rows (target words, trials, train_n).
    Filter via the ``where`` dicts to narrow the view.
    """
    fig, ax = plt.subplots()
    fracs = np.linspace(0.2, 1.0, 5)
    train_cols = [f"m_train_loss_{i}" for i in range(1, 6)]
    val_cols = [f"m_val_loss_{i}" for i in range(1, 6)]

    if not all(c in df.columns for c in train_cols):
        print("plot_loss_curves: training-loss columns missing, skipping.")
        return

    for label, where in lines:
        subset = _filter(df, where).dropna(subset=train_cols, how="all")
        if subset.empty:
            continue
        epochs = int(subset["p_epochs"].iloc[0]) if "p_epochs" in subset.columns else 1
        x = fracs * epochs

        train_means = subset[train_cols].mean().values
        train_stds = subset[train_cols].std().values
        line, = ax.plot(x, train_means, marker="o", label=f"{label} train")
        color = line.get_color()
        if not np.all(np.isnan(train_stds)):
            ax.fill_between(x, train_means - train_stds, train_means + train_stds,
                            alpha=0.15, color=color)

        if all(c in subset.columns for c in val_cols):
            val_means = subset[val_cols].mean().values
            val_stds = subset[val_cols].std().values
            ax.plot(x, val_means, marker="s", linestyle="--",
                    color=color, label=f"{label} val")
            if not np.all(np.isnan(val_stds)):
                ax.fill_between(x, val_means - val_stds, val_means + val_stds,
                                alpha=0.10, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("AE training vs validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_loss_vs_eer(df: pd.DataFrame, lines: list[tuple[str, dict]]):
    """Scatter of final training loss vs EER.

    Shows whether lower reconstruction loss actually translates to better
    anomaly detection. If there is no correlation, the reconstruction
    objective is misaligned with the task.
    """
    if "m_train_loss_5" not in df.columns:
        print("plot_loss_vs_eer: m_train_loss_5 missing, skipping.")
        return

    fig, ax = plt.subplots()
    for label, where in lines:
        subset = _filter(df, where).dropna(subset=["m_train_loss_5"])
        if subset.empty:
            continue
        ax.scatter(subset["m_train_loss_5"], subset["m_eer"], alpha=0.5, s=30, label=label)

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
    sub = sub[sub["p_train_n"] == fixed_train_n] if "p_train_n" in sub.columns else sub
    if sub.empty or "p_covariance_type" not in sub.columns or "p_n_components" not in sub.columns:
        print("plot_gmm_components: no GMM rows in DataFrame, skipping.")
        return

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
    if sub.empty or "p_covariance_type" not in sub.columns or "p_n_components" not in sub.columns:
        print("plot_gmm_diag_vs_full: no GMM rows in DataFrame, skipping.")
        return

    fig, ax = plt.subplots()

    for cov in sorted(sub["p_covariance_type"].unique()):
        for k in sorted(sub["p_n_components"].unique()):
            s = sub[(sub["p_n_components"] == k) & (sub["p_covariance_type"] == cov)]
            label = f"K={k} {cov}"
            _plot_line(ax, s, "p_train_n", y, label)

    ax.set_xlabel("train_n")
    ax.set_ylabel(y.replace("m_", "").upper())
    ax.set_title(f"GMM: {y.replace('m_', '')} by covariance type")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()




def plot_eer_by_target(df: pd.DataFrame, lines: list[tuple[str, dict]],
                        train_n: int, target_label: str = "target",
                        ylabel: str = "EER", y: str = "m_eer"):
    """Grouped bar chart of EER (or any metric) per target class.

    Generalises the per-word / per-subject breakdown: one bar per target class
    per config. Useful to spot which targets are systematically harder to enroll.

    Args:
        df           : results DataFrame
        lines        : list of (label, filter_dict) pairs (one bar group each)
        train_n      : enrollment budget to slice on
        target_label : x-axis label ("word", "subject", ...) — purely cosmetic
        ylabel       : y-axis label
        y            : metric column (default "m_eer")
    """
    sub = df[df["p_train_n"] == train_n]
    if sub.empty:
        print(f"plot_eer_by_target: no rows at train_n={train_n}, skipping.")
        return

    targets = sorted(sub["p_target_class"].unique())
    x = np.arange(len(targets))
    width = 0.8 / max(len(lines), 1)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(targets) + 2), 4))
    for i, (label, where) in enumerate(lines):
        s = _filter(sub, where)
        means = [s.groupby("p_target_class")[y].mean().get(t, float("nan"))
                 for t in targets]
        offset = (i - len(lines) / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in targets], rotation=30, ha="right")
    ax.set_xlabel(target_label)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by {target_label} (train_n={train_n})")
    ax.legend(fontsize=8, ncol=2 if len(lines) > 4 else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def plot_pareto(df: pd.DataFrame, lines: list[tuple[str, dict]],
                x: str = "m_training_flops", y: str = "m_eer",
                train_n: int | None = None):
    """Scatter plot with Pareto-optimal points highlighted per method.

    Shows which methods dominate at different FLOPs budgets. Pareto-optimal
    points (no other point has both lower FLOPs AND lower EER) are highlighted
    with larger markers and connected by lines.

    Args:
        df      : results DataFrame
        lines   : list of (label, filter_dict) pairs
        x       : column for x-axis (default: m_training_flops)
        y       : column for y-axis (default: m_eer, lower is better)
        train_n : if set, restrict to rows with p_train_n == train_n
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    base = df if train_n is None else df[df["p_train_n"] == train_n]

    for label, where in lines:
        subset = _filter(base, where)
        if subset.empty:
            continue

        # Aggregate by unique x value (averaging y across trials AND across
        # distinct configs that share the same x). Configs with identical
        # FLOPs (e.g. GMM K=1 diag and K=1 sph at the same train_n) collapse
        # into one point, so the frontier is computed on those means rather
        # than on each individual config.
        agg = subset.groupby([x])[y].mean().reset_index()
        xs, ys = agg[x].values, agg[y].values

        # Pick a consistent color for this adapter
        color = ax._get_lines.get_next_color()

        # All points as small markers
        ax.scatter(xs, ys, alpha=0.4, s=20, color=color, label=None)

        # Pareto frontier: larger markers + line
        pareto = _pareto_mask(xs, ys)
        if pareto.any():
            px, py = xs[pareto], ys[pareto]
            order = np.argsort(px)
            ax.scatter(px, py, s=80, color=color, label=label, zorder=3)
            ax.plot(px[order], py[order], color=color, linewidth=1.5, alpha=0.7, zorder=2)

    ax.set_xscale("log")
    x_label = x.replace("m_", "").replace("_", " ").title()
    ax.set_xlabel(x_label)
    ax.set_ylabel("EER (lower is better)")
    title = f"Pareto Frontier: EER vs {x_label}"
    if train_n is not None:
        title += f" (train_n={train_n})"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
