import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *

non_seasonal_vars = [
        # Spatial vars
        # "Per-pixel Meltwater Runoff (mmWE/day)", #"ice_velocity",
        # # Non-spatial vars
        # "Air Temperature (C)", "Mélange Area (km^2)", 
        # "Ocean Thermal Forcing (C)", "Terminus Area (km^2)",
        # "Basin-Averaged Meltwater Runoff (mmWE/day)",
        # # Predictor var
        # "Discharge (Gt/yr)",
    ]

def measure_correlation(df: pd.DataFrame):
    # Calculate correlation matrix or load an existing one from s3
    corr_matrix_path = f"{S3_BUCKET}/cjense/data/testmodel/correlation_matrix_{RESOLUTION}{SPECIAL_NAME}.parquet"
    try:
        corr_matrix = pd.read_parquet(corr_matrix_path, storage_options=storage_options)
    except:
        corr_matrix = df.corr(numeric_only=True, method='spearman')
        try:
            corr_matrix.to_parquet(corr_matrix_path, storage_options=storage_options)
        except Exception as e:
            print(f"Warning: could not save correlation matrix to S3: {e}")
    
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='RdBu')
    heatmap.set_title('Correlation Heatmap')
    plt.tight_layout()
    
    plt.savefig('./figures/correlation.png', dpi=300, bbox_inches='tight')
    heatmap_path = f'{FIGURE_DIR}/{GLACIER_NAME}_correlation_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(heatmap_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_correlation_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png')
    plt.clf()
    plt.close()
    
def scatter_matrix(df: pd.DataFrame):
    # Pairwise scatter plots of every variable, with per-season distributions on the diagonal.
    # Points are colored by season so seasonal clusters stand out.
    seasons = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
               6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    plot_df = df.copy()
    plot_df["Season"] = pd.to_datetime(plot_df.index).month.map(seasons)

    grid = sns.pairplot(plot_df, hue="Season", hue_order=["DJF", "MAM", "JJA", "SON"],
                        palette=["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"],
                        plot_kws=dict(s=8, alpha=0.5, edgecolor="none"),
                        diag_kind="kde", diag_kws=dict(common_norm=False, fill=True, alpha=0.2),
                        height=2.2)
    # Wrap the long variable names so neighboring axis labels don't overlap
    for ax in grid.axes.flat:
        ax.set_xlabel(textwrap.fill(ax.get_xlabel(), 20))
        ax.set_ylabel(textwrap.fill(ax.get_ylabel(), 20))
    grid.figure.suptitle(f"Scatter matrix ({RESOLUTION} timestep means)", y=1.01)

    plt.savefig('./figures/scatter_matrix.png', dpi=300, bbox_inches='tight')
    scatter_path = f'{FIGURE_DIR}/{GLACIER_NAME}_scatter_matrix_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(scatter_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_scatter_matrix_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png')
    plt.clf()
    plt.close()

def engineer_features(df: pd.DataFrame, resolution_days: float, feature_cols: list = FEATURE_COLS) -> pd.DataFrame:
    """
    Add lag, rolling, and time-encoding features.
    All lags are in timesteps, not days — adjust shift() values if changing resolution.

    Lag features are critical with one glacier: the model learns entirely from
    space × time variation, so velocity history at each pixel is a strong signal.
    """
    # Lags are taken within each pixel; non-spatial frames have one row per timestep
    pixel_keys = [c for c in ("x", "y") if c in df.columns]
    df = df.sort_values([*pixel_keys, "time"]).reset_index(drop=True)
    px = df.groupby(pixel_keys) if pixel_keys else df

    # ── Lags (in timesteps) ──
    lag_steps = {
        "lag_1step": 1,
        "lag_30d":   max(1, round(30 / resolution_days)),
        "lag_60d":   max(1, round(60 / resolution_days)),
        "lag_90d":   max(1, round(90 / resolution_days)),
    }

    # Only build the lag columns listed in feature_cols (e.g. "lag_60d_airtemp")
    for col in feature_cols:
        for prefix, steps in lag_steps.items():
            if col.startswith(prefix + "_"):
                feature = col[len(prefix) + 1:]
                df[col] = px[feature].shift(steps)

    # ── Time features ──
    t = pd.to_datetime(df["time"])

    # Cyclic seasonality — prevents Dec/Jan discontinuity
    df["season_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25).astype("float32")
    df["season_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25).astype("float32")

    # Long-term trend (0.0 = year 2000, 1.0 = year 2025)
    df["year_norm"] = ((t.dt.year - 2000) / 25).astype("float32")

    # Integer time (days since 2000-01-01) — useful as raw feature too
    df["time_days"] = (t - pd.Timestamp("2000-01-01")).dt.days.astype("int16")

    return df

def load_flat(cache_path: str) -> pd.DataFrame:
    """Per-pixel table: load the cached flat parquet, or build it from the spatial data and cache it."""
    try:
        print(f"Looking for cached flat parquet at {cache_path} ...")
        df = pd.read_parquet(cache_path, storage_options=storage_options)
        print("Found cached file.")

        # Record data size and location
        with open(RESULTS_PATH, 'a') as outfile:
            outfile.write("### Data\n")
            outfile.write(f"Loaded from cache at {cache_path}: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB\n\n")
            df.head().to_markdown(buf=outfile)
            outfile.write("\n\n")

    except Exception:
        print("Did not find cached file. Making new dataframe.")
        
        # Load spatial and non-spatial variables.
        # This method conserves memory by loading the non-spatial variables lazily and merging once
        spatial_df = pd.read_parquet(f's3://gaia/cjense/data/mar/basin_meltwater_regridded_ZI_2000_2025_{RESOLUTION}.parquet', storage_options=storage_options)
        non_spatial = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet", storage_options=storage_options)

        spatial_df = spatial_df.reset_index()
        non_spatial = non_spatial.reset_index()
        non_spatial["time"] = pd.to_datetime(non_spatial["time"]).dt.normalize()
        spatial_df["time"] = pd.to_datetime(spatial_df["time"])

        # Keep only what's needed and downcast to float32 right away to halve memory
        spatial_df = spatial_df[["time", "x", "y", "RU"]].rename(columns={"RU": "Per-pixel Meltwater Runoff (mmWE/day)"})
        spatial_df["Per-pixel Meltwater Runoff (mmWE/day)"] = spatial_df["Per-pixel Meltwater Runoff (mmWE/day)"].astype("float32")
        non_spatial = non_spatial.astype({c: "float32" for c in non_spatial.select_dtypes("float64").columns})

        # Resample both to RESOLUTION with the same bin origin so their timestamps line up.
        # (Spatial data is daily per pixel; without this the lags would be in days, not RESOLUTION steps.)
        origin = pd.Timestamp("2000-01-01")
        spatial_df = (
            spatial_df
            .groupby(["x", "y", pd.Grouper(key="time", freq=RESOLUTION, origin=origin)])["Per-pixel Meltwater Runoff (mmWE/day)"]
            .mean()
            .reset_index()
        )
        non_spatial = non_spatial.resample(RESOLUTION, on="time", origin=origin).mean().reset_index()

        # Left merge: every row needs a pixel; non-spatial values are broadcast to all pixels
        df = spatial_df.merge(non_spatial, on="time", how="left")
        del spatial_df, non_spatial

        # Average timestep length in days for any pandas frequency
        resolution_days = pd.date_range("2000-01-01", periods=101, freq=RESOLUTION).to_series().diff().mean() / pd.Timedelta("1D")
        df = engineer_features(df, resolution_days=resolution_days)

        cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_{RESOLUTION}{SPECIAL_NAME}.parquet"
        print(f"Writing flat parquet to {cache_path} ...")
        df.to_parquet(cache_path, storage_options=storage_options, index=False)
        
        # Record making cached file
        with open(RESULTS_PATH, 'a') as outfile:
            outfile.write("### Data\n")
            outfile.write(f"No cached file found. Made new file at {RESOLUTION} resolution and uploaded it to s3: {cache_path}.\n\n")
            df.head().to_markdown(buf=outfile)
            outfile.write("\n\n")

    return df

def build_nonspatial(feature_cols: list = FEATURE_COLS) -> pd.DataFrame:
    """
    One row per RESOLUTION timestep from the non-spatial parquet only.
    Used when no feature varies by pixel: the per-pixel table would just repeat each
    timestep once per pixel, so this gives the same information in a few MB of memory.
    """
    non_spatial = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet", storage_options=storage_options)
    non_spatial = non_spatial.reset_index()
    non_spatial["time"] = pd.to_datetime(non_spatial["time"]).dt.normalize()
    non_spatial = non_spatial.astype({c: "float32" for c in non_spatial.select_dtypes("float64").columns})

    origin = pd.Timestamp("2000-01-01")
    df = non_spatial.resample(RESOLUTION, on="time", origin=origin).mean().reset_index()

    resolution_days = pd.date_range("2000-01-01", periods=101, freq=RESOLUTION).to_series().diff().mean() / pd.Timedelta("1D")
    return engineer_features(df, resolution_days=resolution_days, feature_cols=feature_cols)

def detrend(df: pd.DataFrame, cols: list, fit_mask: np.ndarray, degree: int = DETREND_DEGREE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove a polynomial-in-time trend from each column in `cols`, in place.

    The trend is fit by least squares on the rows where `fit_mask` is True (the training
    period) and evaluated on every row, so the test period uses the trend extrapolated from
    training. The fitted mean over the training rows is added back, so the result is
    `value - trend(t) + train_mean`: same units and level as the raw data, without the trend.

    Columns that vary by pixel are fit separately per pixel (grouped by x, y when present);
    non-spatial columns have one value per timestep and get a single fit.

    Returns the detrended df and a table of fitted trend slopes (per year, at the start of the period).
    """
    # Time in years since the start of the period (keeps the polynomial well conditioned)
    t = ((pd.to_datetime(df["time"]) - pd.Timestamp(PERIOD_START)).dt.days / 365.25).to_numpy()
    T = np.vander(t, degree + 1, increasing=True)          # columns: 1, t, t^2, ...
    n_coef = degree + 1

    pixel_keys = [c for c in ("x", "y") if c in df.columns]
    group = df.groupby(pixel_keys, sort=False).ngroup().to_numpy() if pixel_keys else np.zeros(len(df), dtype=int)
    n_groups = group.max() + 1

    slopes = {}
    for col in dict.fromkeys(cols):                         # dedupe, keep order
        y = df[col].to_numpy(dtype="float64")
        ok = fit_mask & np.isfinite(y)
        if not ok.any():
            print(f"Warning: no finite training values for '{col}', not detrended.")
            continue

        # Per-group normal equations (T'T) b = T'y, accumulated with bincount so millions
        # of pixel rows don't need a Python loop over pixels
        g = group[ok]
        TT = np.empty((n_groups, n_coef, n_coef))
        Ty = np.empty((n_groups, n_coef))
        for i in range(n_coef):
            Ty[:, i] = np.bincount(g, weights=T[ok, i] * y[ok], minlength=n_groups)
            for j in range(i, n_coef):
                TT[:, i, j] = TT[:, j, i] = np.bincount(g, weights=T[ok, i] * T[ok, j], minlength=n_groups)
        # pinv handles pixels with too few training points (they get a near-zero/flat trend)
        coef = (np.linalg.pinv(TT) @ Ty[..., None])[..., 0]   # (n_groups, n_coef)

        trend = np.einsum("nk,nk->n", T, coef[group])
        n_train = TT[:, 0, 0]
        train_mean = np.divide(Ty[:, 0], n_train, out=np.zeros(n_groups), where=n_train > 0)[group]

        df[col] = (y - trend + train_mean).astype(df[col].dtype)
        slopes[col] = coef[n_train > 0, 1].mean() if degree >= 1 else 0.0

    slope_table = pd.DataFrame({"Trend slope (units/yr)": pd.Series(slopes)})
    return df, slope_table

def plot_detrend(raw_ts: pd.DataFrame, detrended_ts: pd.DataFrame):
    """One panel per variable: raw timestep means, the removed trend, and the detrended series."""
    cols = list(raw_ts.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.4 * len(cols)), sharex=True, squeeze=False)
    for ax, col in zip(axes[:, 0], cols):
        ax.plot(raw_ts.index, raw_ts[col], color="gray", linewidth=1, label="raw")
        ax.plot(detrended_ts.index, detrended_ts[col], color="tab:blue", linewidth=1, label="detrended")
        # Removed trend, shifted to the train mean so it overlays the raw series
        ax.plot(raw_ts.index, raw_ts[col] - detrended_ts[col] + detrended_ts[col].loc[:TRAIN_CUTOFF].mean(),
                color="tab:red", linewidth=1, linestyle="--", label="trend")
        ax.axvline(pd.Timestamp(TRAIN_CUTOFF), color="gray", linestyle=":", linewidth=1)
        ax.set_ylabel(textwrap.fill(col, 22), fontsize=8)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 0].set_title(f"Detrending (degree {DETREND_DEGREE} polynomial fit on {PERIOD_START} to {TRAIN_CUTOFF})")
    plt.tight_layout()

    plt.savefig(f'./figures/detrend_{PERIOD}.png', dpi=300, bbox_inches='tight')
    detrend_path = f'{FIGURE_DIR}/{GLACIER_NAME}_detrend_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.png'
    plt.savefig(detrend_path, dpi=300, bbox_inches='tight')
    fs.put(detrend_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_detrend_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.png')
    plt.close(fig)

def main():
    # First step of the pipeline: start a fresh results file for this period
    with open(RESULTS_PATH, "w") as outfile:
        outfile.write(f"## Period: {PERIOD} (train {PERIOD_START} to {TRAIN_CUTOFF}, test {TEST_START} to {PERIOD_END})\n\n")

    # Look for existing file of the correct resolution
    cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_{RESOLUTION}{SPECIAL_NAME}.parquet"
    if NONSPATIAL_ONLY:
        print("No spatial features in FEATURE_COLS: building one row per timestep from non-spatial data.")
        df = build_nonspatial()
        with open(RESULTS_PATH, 'a') as outfile:
            outfile.write("### Data\n")
            outfile.write(f"Non-spatial run: one row per {RESOLUTION} timestep from {GLACIER_NAME}_non_spatial.parquet "
                          f"({len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.2f} MB)\n\n")
            df.head().to_markdown(buf=outfile)
            outfile.write("\n\n")
    else:
        df = load_flat(cache_path)
    
    # Drop NaNs from target feature
    # You can't predict NaN values!
    df = df.dropna(subset=[TARGET_COL])
    
    # Take the mean at every timestep
    # (keeps non-spatial variables the same, averages spatial variables)
    df_ts = df[["time"] + non_seasonal_vars].groupby("time").mean()
    
    if non_seasonal_vars:
        # Make a correlation heatmap
        measure_correlation(df_ts)

        # Make a scatter matrix
        scatter_matrix(df_ts)
    else:
        print("non_seasonal_vars is empty: skipping correlation heatmap and scatter matrix.")

    # Remove the long-term trend from the target and all physical input features.
    # Done before selecting columns so x/y are still available for per-pixel fits.
    if DETREND:
        df = df[df["time"].between(PERIOD_START, PERIOD_END)].copy()
        detrend_cols = [c for c in [*FEATURE_COLS, TARGET_COL] if c not in DETREND_EXCLUDE]
        fit_mask = (df["time"] < TRAIN_CUTOFF).to_numpy()

        raw_ts = df.groupby("time")[detrend_cols].mean()
        df, slope_table = detrend(df, detrend_cols, fit_mask)
        plot_detrend(raw_ts, df.groupby("time")[detrend_cols].mean())
        del raw_ts

        with open(RESULTS_PATH, 'a') as outfile:
            outfile.write("### Detrending\n")
            outfile.write(f"Removed a degree-{DETREND_DEGREE} polynomial trend (fit on train period "
                          f"{PERIOD_START} to {TRAIN_CUTOFF}, extrapolated over test) from the target and features; "
                          f"train mean added back. Spatial columns fit per pixel (mean slope shown).\n\n")
            outfile.write(slope_table.to_markdown() + "\n\n")

    # Split data into train and test set by time, keeping time alongside the features and target
    # so downstream scripts don't need to split it separately. A random split would leak: the
    # target is the same for every pixel at a timestep, so test timesteps would also appear in train.
    df = df[["time", *FEATURE_COLS, TARGET_COL]]
    train = df[df["time"].between(PERIOD_START, TRAIN_CUTOFF, inclusive="left")]
    test  = df[df["time"].between(TEST_START, PERIOD_END)]
    del df

    # Save train and test sets for run_xgboost.py and run_shap.py
    train.to_parquet(TRAIN_PATH, storage_options=storage_options, index=False)
    test.to_parquet(TEST_PATH, storage_options=storage_options, index=False)
    print(f"Train set saved to S3: {TRAIN_PATH}")
    print(f"Test set saved to S3: {TEST_PATH}")

    return train, test

if __name__ == "__main__":
    train, test = main()
