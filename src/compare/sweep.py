from itertools import product


def sweep(adapter_class: type, param_grid: dict) -> list[tuple[str, type, dict]]:
    """Expand a param grid into (name, class, kwargs) triples.

    Example:
        sweep(GMMAdapter, {"n_components": [1, 3, 5], "covariance_type": ["full", "diag"]})
        → [("GMMAdapter n_components=1 covariance_type=full", GMMAdapter, {...}), ...]
    """
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    results = []
    for vals in combos:
        kwargs = dict(zip(keys, vals))
        tag = " ".join(f"{k}={v}" for k, v in kwargs.items())
        name = f"{adapter_class.__name__} {tag}"
        results.append((name, adapter_class, kwargs))
    return results
