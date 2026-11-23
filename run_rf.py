"""
Random forest on the non-spatial Zachariæ data, with simple hyperparameter tuning and SHAP.

Steps:
  1. Load zach_non_spatial.parquet, average to RESOLUTION, fill short feature gaps by interpolation.
  2. Optionally remove a linear trend from every column (so SHAP explains seasonal/interannual signal).
  3. Tune a small grid with leave-one-year-out CV (each fold holds out one calendar year and
     trains on all the others, before and after it).
  4. Report out-of-fold metrics, fit the final model on all data, and explain it with SHAP.

Runs once per entry in RUNS (different periods / feature sets).
Outputs: modelresults.md and figures/{run}_{predictions,beeswarm,heatmap}.png (picked up by GitHub Actions),
plus SHAP values and predictions saved to S3.

Usage:  python run_rf.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict

########## Settings ##########
DATA_PATH  = "s3://gaia/cjense/data/testmodel/zach_non_spatial.parquet"
OUT_PREFIX = "s3://gaia/cjense/data/testmodel/rf"   # SHAP values + predictions are written under this
RESOLUTION = "MS"                                   # pandas frequency to average to, e.g. "6D", "MS"
SEED       = 42

TARGET = "Discharge (Gt/yr)"#"Near-Terminus Velocity (m/yr)"            # or "Discharge (Gt/yr)"
BASE_FEATURES = [
    "Air Temperature (C)",
    "Ocean Thermal Forcing (C)",
    "Terminus Area (km^2)",
    "Basin-Averaged Meltwater Runoff (mmWE/day)",
]

# Each run is tuned, cross-validated and explained separately. Air temperature starts 2008-07-17;
# mélange area only starts 2015-01-25, so it gets its own shorter run.
RUNS = {
    "2008_2025":         {"start": "2008-07-17", "end": "2025-12-31", "features": BASE_FEATURES},
    "2015_2025_melange": {"start": "2015-01-01", "end": "2025-12-31", "features": BASE_FEATURES + ["Mélange Area (km^2)"]},
}

DETREND = False          # remove a linear trend from target and features (fit over the whole period)

# Small grid: every combination is scored with leave-one-year-out CV
PARAM_GRID = {
    "max_depth":        [None, 10, 5],
    "min_samples_leaf": [1, 5, 10],
    "max_features":     [1.0, 0.5],
}
N_TREES = 300

########## S3 ##########
storage_options = {
    "key": os.environ["AWS_ACCESS_KEY_ID"],
    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    "client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT_URL"]},
    "config_kwargs": {"request_checksum_calculation": "when_required",
                      "response_checksum_validation": "when_required"},
}
fs = s3fs.S3FileSystem(**storage_options)


def load_data(start: str, end: str, features: list) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH, storage_options=storage_options)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")[[*features, TARGET]].resample(RESOLUTION).mean()
    # Sparse features (terminus area, OTF, mélange) have gaps: interpolate inside the record, never extrapolate.
    # Done before cutting to [start, end] so observations just outside the window can anchor the edges.
    df[features] = df[features].interpolate(method="time", limit_area="inside")
    df = df.loc[start:end]
    df = df.dropna()
    print(f"{len(df)} timesteps at {RESOLUTION}, {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
    return df


def detrend(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract a linear fit in time from every column and add the mean back (keeps physical units)."""
    t = (df.index - df.index[0]).days / 365.25
    out = df.copy()
    for col in df.columns:
        slope, intercept = np.polyfit(t, df[col], 1)
        out[col] = df[col] - (slope * t + intercept) + df[col].mean()
    return out


def save_fig(name: str):
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close("all")


def run_model(name: str, start: str, end: str, features: list):
    print(f"\n===== Run {name}: {start} to {end}, {len(features)} features =====")
    raw = load_data(start, end, features)
    df = detrend(raw) if DETREND else raw
    X, y = df[features], df[TARGET]
    years = df.index.year
    cv = LeaveOneGroupOut()

    ##### Tune #####
    search = GridSearchCV(
        RandomForestRegressor(n_estimators=N_TREES, random_state=SEED, n_jobs=-1),
        PARAM_GRID, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    search.fit(X, y, groups=years)
    best = search.best_estimator_
    print(f"Best params: {search.best_params_}  (CV RMSE {-search.best_score_:.4f})")

    ##### Out-of-fold predictions with the best params: each year predicted by a model that never saw it #####
    pred = pd.Series(cross_val_predict(best, X, y, groups=years, cv=cv), index=df.index)
    err = pred - y
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
    by_year = err.groupby(years).apply(lambda e: np.sqrt(np.mean(e ** 2))).rename("RMSE").to_frame()

    ##### Final model on all data, explained with SHAP #####
    best.fit(X, y)
    explainer = shap.TreeExplainer(best)
    sv = explainer(X)

    ##### Figures #####
    label = f"{'Detrended ' if DETREND else ''}{TARGET}"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index, y, color="black", lw=1, label="observed")
    ax.plot(pred.index, pred, color="tab:blue", lw=1, ls="--", label="predicted (year held out)")
    ax.set_ylabel(label)
    ax.set_title(f"Random forest ({name}), leave-one-year-out predictions")
    ax.legend()
    save_fig(f"{name}_predictions")

    shap.plots.beeswarm(sv, show=False, max_display=len(features), plot_size=(12, 6))
    plt.title(f"SHAP values ({name}): {label}")
    save_fig(f"{name}_beeswarm")

    ax = shap.plots.heatmap(sv, instance_order=np.arange(len(X)), max_display=len(features), show=False)
    ax.set_aspect("auto")
    ax.figure.set_size_inches(15, 5)
    first_of_year = np.flatnonzero(pd.Series(years).ne(pd.Series(years).shift()))
    ax.set_xticks(first_of_year)
    ax.set_xticklabels(years[first_of_year], rotation=0)
    ax.set_xlabel("Year")
    ax.set_title(f"SHAP values over time ({name}): {label}")
    save_fig(f"{name}_heatmap")

    ##### Save values to S3 #####
    shap_df = pd.DataFrame(sv.values, index=df.index, columns=[f"shap_{c}" for c in features])
    shap_df["base_value"] = sv.base_values
    shap_df.to_parquet(f"{OUT_PREFIX}_{name}_shap_{RESOLUTION}.parquet", storage_options=storage_options)
    pd.DataFrame({"observed": y, "predicted_oof": pred}).to_parquet(
        f"{OUT_PREFIX}_{name}_predictions_{RESOLUTION}.parquet", storage_options=storage_options)

    ##### Results summary (posted as a GitHub comment by the workflow) #####
    mean_abs_shap = pd.Series(np.abs(sv.values).mean(axis=0), index=features).sort_values(ascending=False)
    with open("modelresults.md", "a") as f:
        f.write(f"## Random forest {name}: {label}\n\n")
        f.write(f"{len(df)} timesteps at {RESOLUTION} ({df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}), "
                f"{'linearly detrended, ' if DETREND else ''}{N_TREES} trees.\n\n")
        f.write("### Leave-one-year-out CV (out-of-fold predictions)\n")
        f.write(f"- RMSE: {rmse:.4f}\n- MAE: {mae:.4f}\n- R²: {r2:.4f}\n\n")
        f.write("Hyperparameters were picked with the same folds, so these are slightly optimistic.\n\n")
        f.write("### Best hyperparameters\n")
        f.writelines(f"- {k}: {v}\n" for k, v in search.best_params_.items())
        f.write("\n### Mean |SHAP| (final model)\n")
        f.write(mean_abs_shap.to_frame("mean |SHAP|").to_markdown() + "\n\n")
        f.write("### RMSE by held-out year\n")
        f.write(by_year.round(4).to_markdown() + "\n\n")
        f.writelines(f"![{name}_{fig}](FIGURE_URL/{name}_{fig}.png)\n" for fig in ("predictions", "beeswarm", "heatmap"))
        f.write("\n")


def main():
    open("modelresults.md", "w").close()    # fresh results file; each run appends its section
    for name, run in RUNS.items():
        run_model(name, run["start"], run["end"], run["features"])
    print(open("modelresults.md").read())


if __name__ == "__main__":
    main()
