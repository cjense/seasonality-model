import numpy as np
import pandas as pd

from config import *

def engineer_features(df: pd.DataFrame, resolution_days: float) -> pd.DataFrame:
    """
    Add lag, rolling, and time-encoding features.
    All lags are in timesteps, not days — adjust shift() values if changing resolution.

    Lag features are critical with one glacier: the model learns entirely from
    space × time variation, so velocity history at each pixel is a strong signal.
    """
    df = df.sort_values(["x", "y", "time"]).reset_index(drop=True)
    px = df.groupby(["x", "y"])

    # ── Velocity lags (in timesteps) ──
    steps_30d  = max(1, round(30  / resolution_days))
    steps_60d  = max(1, round(60  / resolution_days))
    steps_90d  = max(1, round(90  / resolution_days))
    
    non_seasonal_vars = [
        # Spatial vars
        "meltwater", #"ice_velocity",
        # Non-spatial vars
        "airtemp", "masked_mel_velocity", "melange_area_km2", 
        "ocean_EN4_TFc", "area_km2", "area_change_km2",
        # Space
        "Discharge (Gt/yr)",
    ]

    for feature in non_seasonal_vars:
        df["lag_1step_"+feature] = px[feature].shift(1)
        df["lag_30d_"+feature]   = px[feature].shift(steps_30d)
        df["lag_60d_"+feature]   = px[feature].shift(steps_60d)
        df["lag_90d_"+feature]   = px[feature].shift(steps_90d)

        # Rolling mean over past ~30 days (excludes current timestep via shift first)
        df["roll_30d_mean_"+feature] = (
            px[feature]
            .transform(lambda s: s.shift(1).rolling(steps_30d, min_periods=1).mean())
        )
        df["roll_30d_std_"+feature] = (
            px[feature]
            .transform(lambda s: s.shift(1).rolling(steps_30d, min_periods=1).std())
        )

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
    # First step of the pipeline: start a fresh modelresults.md for this run
    open("modelresults.md", "w").close()

    # Look for existing file of the correct resolution
    cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_{RESOLUTION}{SPECIAL_NAME}.parquet"
    try:
        print(f"Looking for cached flat parquet at {cache_path} ...")
        df = pd.read_parquet(cache_path, storage_options=storage_options)
        print("Found cached file.")

        # Record data size and location
        with open("modelresults.md", 'a') as outfile:
            outfile.write("### Data\n")
            outfile.write(f"Loaded from cache at {cache_path}: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB\n\n")
            df.head().to_markdown(buf=outfile)
            outfile.write("\n\n")

    except Exception:
        print("Did not find cached file. Making new dataframe.")
        
        # Load spatial and non-spatial variables.
        # This method conserves memory by loading the non-spatial variables lazily and merging once
        spatial_df = pd.read_parquet(f's3://gaia/cjense/data/mar/basin_meltwater_regridded_daily_ZI_2000_2025_{RESOLUTION}.parquet', storage_options=storage_options)
        non_spatial = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet", storage_options=storage_options)

        spatial_df = spatial_df.reset_index()
        non_spatial = non_spatial.reset_index()
        non_spatial["time"] = pd.to_datetime(non_spatial["time"]).dt.normalize()
        spatial_df["time"] = pd.to_datetime(spatial_df["time"])

        spatial_df = spatial_df.rename(columns={"RU": "meltwater"})
        non_spatial = non_spatial.resample(RESOLUTION, on="time").mean().reset_index()
        df = spatial_df.merge(non_spatial, on="time", how="outer")
        df = df.astype({c: "float32" for c in df.select_dtypes("float64").columns})  # halve memory

        # Average timestep length in days for any pandas frequency
        resolution_days = pd.date_range("2000-01-01", periods=101, freq=RESOLUTION).to_series().diff().mean() / pd.Timedelta("1D")
        df = engineer_features(df, resolution_days=resolution_days)

        # TODO: Optimize datatypes (convert to float32) to conserve memory

        cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_{RESOLUTION}{SPECIAL_NAME}.parquet"
        print(f"Writing flat parquet to {cache_path} ...")
        df.to_parquet(cache_path, storage_options=storage_options, index=False)
        
        # Record making cached file
        with open("modelresults.md", 'a') as outfile:
            outfile.write("### Data\n")
            outfile.write(f"No cached file found. Made new file at {RESOLUTION} resolution and uploaded it to s3: {cache_path}.\n\n")
            df.head().to_markdown(buf=outfile)
            outfile.write("\n\n")
    
    # Drop NaNs from target feature
    # You can't predict NaN values!
    df = df.dropna(subset=[TARGET_COL])

    # Split data into train and test set by time, keeping time alongside the features and target
    # so downstream scripts don't need to split it separately. A random split would leak: the
    # target is the same for every pixel at a timestep, so test timesteps would also appear in train.
    df = df[["time", *FEATURE_COLS, TARGET_COL]]
    train = df[df["time"] < TRAIN_CUTOFF]
    test  = df[df["time"] >= TEST_START]

    # Save train and test sets for run_xgboost.py and run_shap.py
    train.to_parquet(TRAIN_PATH, storage_options=storage_options, index=False)
    test.to_parquet(TEST_PATH, storage_options=storage_options, index=False)
    print(f"Train set saved to S3: {TRAIN_PATH}")
    print(f"Test set saved to S3: {TEST_PATH}")

    return train, test

if __name__ == "__main__":
    train, test = main()
