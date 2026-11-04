import os
import textwrap
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from geoshapley import GeoShapleyTreeExplainer

from config import *
from geoshapley_fast import explain_fast

GEO_COLS = ["x", "y"]                 # GeoShapley treats the last g columns as one GEO player
NON_GEO = FEATURE_COLS[:-len(GEO_COLS)]
FIG_DIR = "/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures"
N_MAP_TERMS = 6                       # number of feature x GEO interaction maps to plot


def sample_timesteps(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Pick random whole timesteps (every pixel at each chosen time) up to GEOSHAP_MAX_ROWS rows,
    so every pixel in the maps is averaged over the same set of times.
    '''
    n_pixels = df.groupby(GEO_COLS).ngroups
    times = np.sort(df["time"].unique())
    n_times = max(1, min(len(times), GEOSHAP_MAX_ROWS // n_pixels))
    rng = np.random.default_rng(MODEL_SEED)
    chosen = rng.choice(times, size=n_times, replace=False)
    sample = df[df["time"].isin(chosen)].reset_index(drop=True)
    print(f"Explaining {n_times} of {len(times)} timesteps x {n_pixels:,} pixels = {len(sample):,} rows")
    return sample


def validate(model: xgb.Booster, X: pd.DataFrame, base, primary, geo, geo_intera, n_lib=25, n_shap=20_000):
    '''
    Check the GeoShapley values three ways; raises if either of the first two fails.
      1. Additivity: base + primary + GEO + interactions must equal XGBoost's own prediction.
      2. The vectorized code must reproduce geoshapley's GeoShapleyTreeExplainer exactly
         (on NaN-free rows -- the library mishandles NaNs, see note below).
      3. Converted to ordinary Shapley values, they should be close to XGBoost's TreeSHAP
         (not identical: TreeSHAP treats x and y as two players, GeoShapley as one).
    '''
    lines = []

    # 1. Additivity against the real model
    pred = model.predict(xgb.DMatrix(X))
    total = base + primary.sum(axis=1) + geo + geo_intera.sum(axis=1)
    err = np.abs(total - pred).max()
    tol = 1e-4 * max(1.0, np.abs(pred).max())   # XGBoost predicts in float32
    lines.append(f"Additivity vs XGBoost prediction: max |error| = {err:.2e} (tolerance {tol:.1e})")
    if err > tol:
        raise RuntimeError(lines[-1])

    # 2. Match the reference implementation.
    # geoshapley 0.2.x's _Split.matches returns False for NaN before checking XGBoost's
    # default (missing) branch, so the library is wrong for rows with NaNs (e.g. early lags).
    clean = X.dropna()
    idx = clean.index[:n_lib]
    ref = GeoShapleyTreeExplainer(model, g=len(GEO_COLS)).explain(X.loc[idx])
    lib_err = max(np.abs(primary[idx] - ref.primary).max(),
                  np.abs(geo[idx] - ref.geo).max(),
                  np.abs(geo_intera[idx] - ref.geo_intera).max())
    lines.append(f"Vs geoshapley GeoShapleyTreeExplainer on {len(idx)} NaN-free rows: max |diff| = {lib_err:.2e}")
    if lib_err > tol:
        raise RuntimeError(lines[-1])

    # 3. Compare with TreeSHAP (informational)
    rows = np.random.default_rng(MODEL_SEED).choice(len(X), size=min(n_shap, len(X)), replace=False)
    contribs = model.predict(xgb.DMatrix(X.iloc[rows]), pred_contribs=True)   # last column = bias
    k = len(NON_GEO)
    shap_feat = primary[rows] + geo_intera[rows] / 2       # GeoShapleyResults.geoshap_to_shap()
    shap_geo = geo[rows] + geo_intera[rows].sum(axis=1) / 2
    shap_xy = contribs[:, k] + contribs[:, k + 1]

    def corr(a, b):
        return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

    r_feat = [corr(shap_feat[:, j], contribs[:, j]) for j in range(k)]
    lines.append(f"Vs XGBoost TreeSHAP on {len(rows):,} rows: median feature correlation = {np.nanmedian(r_feat):.3f}, "
                 f"GEO vs SHAP(x)+SHAP(y) correlation = {corr(shap_geo, shap_xy):.3f}")
    # GeoShapley's projection moves part of feature-feature interactions onto GEO terms, so they are
    # nonzero even for a model that ignores x/y. TreeSHAP of x and y shows how much location is really used.
    lines.append(f"Mean |GEO + interactions/2| = {np.abs(shap_geo).mean():.3g}; "
                 f"mean |SHAP(x) + SHAP(y)| = {np.abs(shap_xy).mean():.3g}; "
                 f"mean |prediction - base| = {np.abs(pred[rows] - base).mean():.3g} ({TARGET_COL})")

    n_nan_rows = len(X) - len(clean)
    lines.append(f"Rows with at least one NaN feature: {n_nan_rows:,} of {len(X):,}")
    for line in lines:
        print(line)
    return lines


def to_dataframe(sample: pd.DataFrame, base, primary, geo, geo_intera) -> pd.DataFrame:
    df = pd.DataFrame(primary, columns=[f"primary_{c}" for c in NON_GEO])
    df.insert(0, "time", sample["time"].to_numpy())
    df.insert(1, "x", sample["x"].to_numpy())
    df.insert(2, "y", sample["y"].to_numpy())
    df["GEO"] = geo
    df[[f"{c} x GEO" for c in NON_GEO]] = geo_intera
    df["base_value"] = base
    return df


def save_figure(fig, name: str):
    fname = f"{GLACIER_NAME}_{name}_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.png"
    fig.savefig(f"./figures/{name}_{PERIOD}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=300, bbox_inches="tight")
    fs.put(f"{FIG_DIR}/{fname}", f"{S3_BUCKET}/cjense/data/testmodel/figures/{fname}")
    plt.close(fig)


def summary_plot(X: pd.DataFrame, primary, geo, geo_intera):
    '''Beeswarm of primary, GEO and feature x GEO values (same layout as GeoShapleyResults.summary_plot).'''
    rng = np.random.default_rng(MODEL_SEED)
    rows = np.sort(rng.choice(len(X), size=min(len(X), BEESWARM_MAX_POINTS), replace=False))
    values = np.hstack((primary[rows], geo[rows, None], geo_intera[rows]))
    data = X[NON_GEO].iloc[rows].reset_index(drop=True)
    data["GEO"] = 0
    data[[f"{c} x GEO" for c in NON_GEO]] = X[NON_GEO].iloc[rows].to_numpy()

    shap.summary_plot(values, data, show=False, max_display=20, plot_size=(14, 10))
    plt.gca().set_xlabel(f"GeoShapley value (impact on predicted {TARGET_COL})")
    plt.title(f"GeoShapley values ({PERIOD}, {len(rows):,} rows)")
    save_figure(plt.gcf(), "geoshapley_summary")


def spatial_map(df_geo: pd.DataFrame):
    '''
    Map per-pixel GeoShapley values, averaged over the sampled timesteps:
      - GEO: intrinsic location effect (signed mean)
      - Location Shapley value = GEO + half of every feature x GEO interaction
        (GeoShapleyResults.geoshap_to_shap); exactly 0 if the model ignores x/y (signed mean)
      - top feature x GEO interactions: mean |value|, since their sign flips with the feature
        value through the season and a signed mean would cancel out
    Note GEO and the x GEO terms on their own can be nonzero even when the model never uses
    x/y: GeoShapley's projection moves part of feature-feature interactions onto them.
    '''
    inter_cols = [f"{c} x GEO" for c in NON_GEO]
    loc = "Location Shapley value\n(GEO + ½ Σ interactions)"
    df_geo = df_geo.assign(**{loc: df_geo["GEO"] + df_geo[inter_cols].sum(axis=1) / 2})
    top = df_geo[inter_cols].abs().mean().sort_values(ascending=False).index[:N_MAP_TERMS].tolist()

    signed = df_geo.groupby(GEO_COLS)[["GEO", loc]].mean()
    magnitude = df_geo[GEO_COLS + top].assign(**{c: df_geo[c].abs() for c in top}).groupby(GEO_COLS).mean()
    px = signed.join(magnitude).reset_index()

    panels = [("GEO", "signed"), (loc, "signed")] + [(c, "abs") for c in top]
    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False, layout="constrained")
    # Signed panels share one symmetric scale, so a ~0 location map shows as white, not amplified noise
    signed_lim = np.nanpercentile(np.abs(px[["GEO", loc]].to_numpy()), 99) or 1e-12
    for ax, (col, kind) in zip(axes.flat, panels):
        grid = px.pivot(index="y", columns="x", values=col)
        vals = grid.to_numpy()
        if kind == "signed":
            lim = signed_lim
            kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
            label = "mean value (Gt/yr)"
        else:
            kw = dict(cmap="viridis", vmin=0, vmax=np.nanpercentile(vals, 99))
            label = "mean |value| (Gt/yr)"
        mesh = ax.pcolormesh(grid.columns / 1000, grid.index / 1000, vals, shading="nearest", **kw)
        fig.colorbar(mesh, ax=ax, label=label, shrink=0.85)
        title = col if "\n" in col else textwrap.fill(col.replace(" x GEO", ""), 34) + ("\n× GEO" if col.endswith(" x GEO") else "")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    n_times = df_geo["time"].nunique()
    fig.suptitle(f"{GLACIER_NAME} GeoShapley values per pixel ({PERIOD}, mean over {n_times} sampled timesteps)")
    save_figure(fig, "geoshapley_map")
    return px


def main():
    # Load the train set written by preprocess.py
    train = pd.read_parquet(TRAIN_PATH, storage_options=storage_options)

    # Load the model written by run_xgboost.py
    model = xgb.Booster()
    try:
        fs.get(S3_MODEL_PATH, LOCAL_MODEL_PATH)
        print("Model found in S3 bucket.")
    except Exception:
        print("Model not found in S3. Loading from disk.")
    model.load_model(LOCAL_MODEL_PATH)
    model.set_param({"device": "cpu"})

    # GeoShapley groups the last g columns into the GEO player, and maps XGBoost splits by name
    assert FEATURE_COLS[-len(GEO_COLS):] == GEO_COLS, "x and y must be the last FEATURE_COLS"
    assert model.feature_names == FEATURE_COLS, "model features don't match FEATURE_COLS -- retrain the model"

    sample = sample_timesteps(train)
    X = sample[FEATURE_COLS]

    print("Calculating GeoShapley values...")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    base, primary, geo, geo_intera = explain_fast(model, X, g=len(GEO_COLS), n_jobs=n_jobs)
    checks = validate(model, X, base, primary, geo, geo_intera)

    # Save values keyed by (time, x, y) so they can be joined back to the feature parquet
    df_geo = to_dataframe(sample, base, primary, geo, geo_intera)
    save_path = f"{S3_BUCKET}/cjense/data/testmodel/geoshapley_values_{GLACIER_NAME}_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.parquet"
    df_geo.to_parquet(save_path, storage_options=storage_options, index=False)
    print(f"GeoShapley value parquet saved to S3: {save_path}")

    summary_plot(X, primary, geo, geo_intera)
    spatial_map(df_geo)

    with open(RESULTS_PATH, "a") as outfile:
        outfile.write("### GeoShapley\n")
        outfile.write(f"Values for {len(df_geo):,} train rows ({df_geo['time'].nunique()} timesteps), saved to {save_path}\n\n")
        for line in checks:
            outfile.write(f"- {line}\n")
        outfile.write("\n")

    return model, df_geo


if __name__ == "__main__":
    model, df_geo = main()
