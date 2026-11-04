import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *

non_seasonal_vars = [
        # Spatial vars
        "meltwater", #"ice_velocity",
        # Non-spatial vars
        "airtemp", "masked_mel_velocity", "melange_area_km2", 
        "ocean_EN4_TFc", "area_km2",
        # Space
        "Discharge (Gt/yr)",
    ]

def measure_correlation(df: pd.DataFrame):
    # Take the mean at every timestep
    # (keeps non-spatial variables the same, averages spatial variables)
    df_ts = df[["time"] + non_seasonal_vars].groupby("time").mean()
    
    # Calculate correlation matrix or load an existing one from s3
    corr_matrix_path = f"{S3_BUCKET}/cjense/data/testmodel/correlation_matrix_{RESOLUTION}{SPECIAL_NAME}.parquet"
    try:
        corr_matrix = pd.read_parquet(corr_matrix_path, storage_options=storage_options)
    except:
        corr_matrix = df_ts.corr(numeric_only=True, method='spearman')
        try:
            corr_matrix.to_parquet(corr_matrix_path, storage_options=storage_options)
        except Exception as e:
            print(f"Warning: could not save correlation matrix to S3: {e}")
    
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap')
    plt.tight_layout()
    
    plt.savefig('./figures/correlation.png', dpi=300, bbox_inches='tight')
    heatmap_path = f'/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures/{GLACIER_NAME}_correlation_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(heatmap_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_correlation_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png')
    plt.clf()
    plt.close()

def engineer_features(df: pd.DataFrame, resolution_days: float) -> pd.DataFrame:
    """
    Add lag, rolling, and time-encoding features.
    All lags are in timesteps, not days — adjust shift() values if changing resolution.

    Lag features are critical with one glacier: the model learns entirely from
    space × time variation, so velocity history at each pixel is a strong signal.
    """
    df = df.sort_values(["x", "y", "time"]).reset_index(drop=True)
    px = df.groupby(["x", "y"])

    # ── Lags (in timesteps) ──
    lag_steps = {
        "lag_1step": 1,
        "lag_30d":   max(1, round(30 / resolution_days)),
        "lag_60d":   max(1, round(60 / resolution_days)),
        "lag_90d":   max(1, round(90 / resolution_days)),
    }

    # Only build the lag columns listed in FEATURE_COLS (e.g. "lag_60d_airtemp")
    for col in FEATURE_COLS:
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

def main():
    # First step of the pipeline: start a fresh results file for this period
    with open(RESULTS_PATH, "w") as outfile:
        outfile.write(f"## Period: {PERIOD} (train {PERIOD_START} to {TRAIN_CUTOFF}, test {TEST_START} to {PERIOD_END})\n\n")

    # Look for existing file of the correct resolution
    cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_{RESOLUTION}{SPECIAL_NAME}.parquet"
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
        spatial_df = pd.read_parquet(f's3://gaia/cjense/data/mar/basin_meltwater_regridded_daily_ZI_2000_2025.parquet', storage_options=storage_options)
        non_spatial = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet", storage_options=storage_options)

        spatial_df = spatial_df.reset_index()
        non_spatial = non_spatial.reset_index()
        non_spatial["time"] = pd.to_datetime(non_spatial["time"]).dt.normalize()
        spatial_df["time"] = pd.to_datetime(spatial_df["time"])

        # Keep only what's needed and downcast to float32 right away to halve memory
        spatial_df = spatial_df[["time", "x", "y", "RU"]].rename(columns={"RU": "meltwater"})
        spatial_df["meltwater"] = spatial_df["meltwater"].astype("float32")
        non_spatial = non_spatial.astype({c: "float32" for c in non_spatial.select_dtypes("float64").columns})

        # Resample both to RESOLUTION with the same bin origin so their timestamps line up.
        # (Spatial data is daily per pixel; without this the lags would be in days, not RESOLUTION steps.)
        origin = pd.Timestamp("2000-01-01")
        spatial_df = (
            spatial_df
            .groupby(["x", "y", pd.Grouper(key="time", freq=RESOLUTION, origin=origin)])["meltwater"]
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
    
    # Drop NaNs from target feature
    # You can't predict NaN values!
    df = df.dropna(subset=[TARGET_COL])
    
    # Make a correlation heatmap
    measure_correlation(df)

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
