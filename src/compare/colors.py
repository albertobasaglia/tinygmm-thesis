"""Single source of truth for figure colors.

Every adaptive layer must get the SAME color in every figure it appears in --
validation, held-out test, Pareto frontiers, CI bars, the on-device benchmark,
and the resource charts -- regardless of which other series happen to share the
axes. Matplotlib's default color cycle does not give that: it assigns colors by
plotting order, so an algorithm drifts to a different color whenever the set (or
order) of lines changes between figures. Routing every plot through the helpers
here removes that drift.

Nothing else in src/compare/ should hard-code an algorithm color: import from
this module instead. Two entry points cover all callers:

  * ``color_for(where)``       -- structured: a ``p_``-prefixed filter dict (the
                                  comparison/Pareto plots already carry these).
  * ``color_for_label(label)`` -- heuristic: a display string (the benchmark and
                                  resource charts only have a label to go on).

Both resolve to the same FAMILY/COV tables, so the two paths never disagree.
"""

# --- Algorithm identity (used across every comparison figure) ------------------

# One base color per adaptive-layer family. Keyed by the adapter class name so a
# ``where`` dict ({"p_adapter": ...}) maps straight through.
FAMILY = {
    "GMMAdapter":       "#4C72B0",  # blue   (the thesis focus)
    "KNNAdapter":       "#55A868",  # green
    "SmallAEAdapter":   "#DD8452",  # orange
    "CosineAdapter":    "#C44E52",  # red
    "PrototypeAdapter": "#8172B2",  # purple
}

# GMM variants are shades of the GMM blue, keyed by covariance type, so the two
# frozen GMM rows (K=1 full / K=1 diag) and the per-covariance GMM benchmark
# families stay distinct-but-related everywhere they appear. The medium shade is
# the GMM family base, so a GMM plotted at the family level (no covariance given)
# matches its diag variant.
COV = {
    "spherical": "#9ECAE1",  # light blue
    "diag":      "#4C72B0",  # medium blue (== FAMILY["GMMAdapter"])
    "full":      "#0B2545",  # dark navy -- kept well clear of diag in lightness
                             # so the two GMM variants stay distinct even as
                             # small Pareto/CI markers
}

# Curves that are not an algorithm comparison (e.g. the extractor's train vs
# validation loss). Kept here so no hex lives outside this module.
LEARNING_CURVE = {"train": "#4C72B0", "val": "#DD8452"}

FALLBACK = "#7F7F7F"  # gray, for any adapter without an assigned color


def color_for(where: dict) -> str:
    """Identity color for a comparison line given its ``p_``-prefixed filter.

    GMM lines are colored by covariance type when one is specified (so K=1 full
    and K=1 diag differ); every other family uses its base color. Because the
    result depends only on the line's own identity -- never on which other series
    share the axes -- an algorithm keeps its color across all figures.
    """
    adapter = where.get("p_adapter")
    if adapter == "GMMAdapter":
        cov = where.get("p_covariance_type")
        if cov in COV:
            return COV[cov]
        return FAMILY["GMMAdapter"]
    return FAMILY.get(adapter, FALLBACK)


def color_for_label(label: str) -> str:
    """Identity color from a display label, for callers that only have a string.

    Mirrors ``color_for`` for the benchmark and resource charts, whose series
    are named ("GMM diag", "kNN k=5", "SmallAE", ...) rather than carried as
    filter dicts. Matching is by keyword so the various phrasings of the same
    family ("AE"/"SmallAE", "GMM"/"GMM full"/"GMM K=1 full") all resolve alike.
    """
    low = label.lower()
    if "gmm" in low:
        if "full" in low:
            return COV["full"]
        if "diag" in low:
            return COV["diag"]
        if "spher" in low or "sph" in low:
            return COV["spherical"]
        return FAMILY["GMMAdapter"]
    if "knn" in low or "k-nn" in low or "nearest" in low:
        return FAMILY["KNNAdapter"]
    if "cosine" in low:
        return FAMILY["CosineAdapter"]
    if "prototype" in low:
        return FAMILY["PrototypeAdapter"]
    if "ae" in low:  # AE / SmallAE / autoencoder -- checked last (e.g. after kNN)
        return FAMILY["SmallAEAdapter"]
    return FALLBACK


# --- Within-figure ablation palette (order-stable) -----------------------------
# Series such as the nine GMM K x covariance lines or the AE latent-dim lines
# appear only inside their own ablation figure, never alongside the comparison
# families, so cross-figure identity does not apply. An ordered palette is enough:
# the line list and its order are fixed, so colors stay stable across the EER /
# ACC / CI variants of the same ablation.
ABLATION = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def cycle(n: int) -> list[str]:
    """First ``n`` ablation colors (wraps if ``n`` exceeds the palette length)."""
    return [ABLATION[i % len(ABLATION)] for i in range(n)]
