"""
Row-vectorized version of geoshapley's exact tree-path GeoShapley explainer.

geoshapley.GeoShapleyTreeExplainer computes the exact values one row at a time in pure
Python (~1 s/row for a 1000-tree XGBoost model), which is far too slow for millions of
pixel-timesteps. This module runs the *same* algorithm (geoshapley_tree._tree_geoshap_values)
but loops over tree paths once and does the per-row arithmetic with numpy across all rows.

The tree paths, split directions, branch (cover) probabilities and base value come straight
from the library's own adapter, so the only new code is the vectorized arithmetic.
run_geoshapley.py checks it against the library on a subset of rows.
"""
import numpy as np
from geoshapley.geoshapley_tree import _make_tree_adapter


def _matches(split, X, X32):
    """Vectorized geoshapley_tree._Split.matches over all rows of one column."""
    v = X[:, split.feature]
    nan = np.isnan(v)
    d = split.direction
    if d == "missing":
        return nan
    if d in ("lt", "ge", "lt_missing", "ge_missing"):
        # XGBoost compares in float32
        v32, t32 = X32[:, split.feature], np.float32(split.threshold)
        with np.errstate(invalid="ignore"):
            m = v32 < t32 if d.startswith("lt") else v32 >= t32
    elif d in ("le", "gt", "le_missing", "gt_missing"):
        t = split.threshold
        with np.errstate(invalid="ignore"):
            m = v <= t if d.startswith("le") else v > t
    else:
        raise ValueError(f"Unknown split direction: {d}")
    return (m | nan) if d.endswith("_missing") else (m & ~nan)


def _path_geoshap(path, X, X32, k, primary, geo, geo_intera):
    """Add one leaf path's GeoShapley terms for all rows (in place)."""
    n = X.shape[0]
    factors = {}          # player -> (n,) factor; insertion order matches the library
    base_weight = 1.0
    for split in path["splits"]:
        player = split.feature if split.feature < k else k
        base_weight *= split.probability
        f = factors.get(player)
        if f is None:
            f = np.ones(n)
        # library: once a player's factor is 0 it stays 0; otherwise /prob on match, else 0
        factors[player] = f * _matches(split, X, X32) / split.probability

    scale = path["value"] * base_weight
    feature_ids = [p for p in factors if p < k]
    deltas = [factors[p] - 1.0 for p in feature_ids]
    geo_delta = factors[k] - 1.0 if k in factors else 0.0

    # e[q] = q-th elementary symmetric polynomial of the deltas, per row
    m = len(deltas)
    e = [np.ones(n)] + [np.zeros(n) for _ in range(m)]
    for degree, delta in enumerate(deltas, start=1):
        for q in range(degree, 0, -1):
            e[q] = e[q] + delta * e[q - 1]

    for q in range(1, m + 1):
        geo -= scale * ((q - 1.0) / (q + 1.0)) * e[q]
    for q in range(m + 1):
        geo += scale * geo_delta * ((1.0 - q) / (q + 1.0)) * e[q]

    for feature, delta in zip(feature_ids, deltas):
        # Divide prod_j (1 + delta_j t) by (1 + delta t)
        excluded = [np.ones(n)]
        for q in range(1, m):
            excluded.append(e[q] - delta * excluded[q - 1])
        for q in range(1, m + 1):
            included = delta * excluded[q - 1]
            primary[:, feature] += scale * (2.0 / (q * (q + 1.0))) * included
            geo_intera[:, feature] += (
                scale * (2.0 * (q - 1.0) / (q * (q + 1.0))) * included
                + scale * geo_delta * (2.0 / (q + 1.0)) * included
            )


def _explain_chunk(adapter, X, k):
    X32 = X.astype(np.float32)
    primary = np.zeros((X.shape[0], k))
    geo = np.zeros(X.shape[0])
    geo_intera = np.zeros((X.shape[0], k))
    for tree in adapter.trees:
        for path in tree["paths"]:
            if tree["weight"] != 1.0:
                path = {"splits": path["splits"], "value": path["value"] * tree["weight"]}
            _path_geoshap(path, X, X32, k, primary, geo, geo_intera)
    return primary, geo, geo_intera


def _explain_chunk_xgb(model_raw, feature_names, X, k):
    """Worker for process pools: rebuild the XGBoost booster and its paths in the child."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(model_raw)
    booster.feature_names = feature_names
    return _explain_chunk(_make_tree_adapter(booster, feature_names=feature_names), X, k)


def explain_fast(model, X_geo, g=2, chunk_size=50_000, n_jobs=1, verbose=True):
    """
    Exact tree-path GeoShapley values for every row of X_geo (geo columns last).

    Returns (base_value, primary (n,k), geo (n,), geo_intera (n,k)) -- the same quantities
    as GeoShapleyTreeExplainer(model, g).explain(X_geo).
    n_jobs > 1 splits rows across processes (native xgboost.Booster models only).
    """
    feature_names = list(X_geo.columns) if hasattr(X_geo, "columns") else None
    X_all = np.asarray(X_geo, dtype=np.float64)
    n, M = X_all.shape
    k = M - g
    adapter = _make_tree_adapter(model, feature_names=feature_names)

    primary = np.zeros((n, k))
    geo = np.zeros(n)
    geo_intera = np.zeros((n, k))

    parallel = n_jobs > 1 and hasattr(model, "save_raw")
    if parallel:
        chunk_size = min(chunk_size, -(-n // n_jobs))
    starts = list(range(0, n, chunk_size))

    def store(c, result):
        sl = slice(starts[c], starts[c] + chunk_size)
        primary[sl], geo[sl], geo_intera[sl] = result
        if verbose:
            print(f"  GeoShapley chunk {c + 1}/{len(starts)} done", flush=True)

    if parallel:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        model_raw = model.save_raw(raw_format="ubj")
        with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_explain_chunk_xgb, model_raw, feature_names,
                                   X_all[s:s + chunk_size], k): c for c, s in enumerate(starts)}
            for fut in as_completed(futures):
                store(futures[fut], fut.result())
    else:
        for c, s in enumerate(starts):
            store(c, _explain_chunk(adapter, X_all[s:s + chunk_size], k))

    return adapter.empty_value(), primary, geo, geo_intera
