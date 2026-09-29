import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import xgboost as xgb
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import s3fs
import os
import shap
import matplotlib.pyplot as plt
from geoshapley import GeoShapleyTreeExplainer

########## CONFIG ##########
GLACIER_NAME  = "zach"
S3_BUCKET     = "s3://gaia"
RESOLUTION    = "1D"
TRAIN_CUTOFF  = "2022-01-01"          # everything before this is train
TEST_START    = "2022-01-01"          # everything from here is test
CACHE_PARQUET = True                  # write flat df to S3 after extraction
MODEL_SEED    = 42                    # random seed to train model
SPLIT_SEED    = 123                   # random seed to split train and test data
SHAP_PLOT_SAMPLES = 20_000            # rows drawn for beeswarm/heatmap plots

FEATURE_COLS = [
    # Spatial vars
    "meltwater", "ice_velocity",                                            # TODO: Add ice_elevation, distance_to_terminus
    # Non-spatial vars (broadcast)
    "airtemp", "masked_mel_velocity", "melange_area_km", "ocean_EN4_TFc",   # TODO: Add tongue_length, average_meltwater_runoff, melange_rigidity
    # Lag features
    "lag_1step", "lag_30d", "lag_60d", "lag_90d",
    "roll_30d_mean", "roll_30d_std",
    # Time (encoded features only — raw datetime kept separate)
    "season_sin", "season_cos", "year_norm", "time_days",
    # Space
    "x", "y",
]

NON_SEASONAL_VARS = [
    # Spatial vars
    "meltwater", "ice_velocity",
    # Non-spatial vars
    "airtemp", "masked_mel_velocity", "melange_area_km", "ocean_EN4_TFc",
    # Space
    "x", "y",
]

TARGET_COL = "discharge"

########## S3 Setup ##########
fs = s3fs.S3FileSystem(
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    client_kwargs={"endpoint_url": os.environ["S3_ENDPOINT_URL"]},
    config_kwargs={
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
                }
)

storage_options = {
    "client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT_URL"]},
    "config_kwargs": {
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
    },
}

def engineer_features(df: pd.DataFrame, resolution_days: int = 6) -> pd.DataFrame:
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

    df["lag_1step"] = px["discharge"].shift(1)          # 1 timestep ago
    df["lag_30d"]   = px["discharge"].shift(steps_30d)
    df["lag_60d"]   = px["discharge"].shift(steps_60d)
    df["lag_90d"]   = px["discharge"].shift(steps_90d)

    # Rolling mean over past ~30 days (excludes current timestep via shift first)
    df["roll_30d_mean"] = (
        px["discharge"]
        .transform(lambda s: s.shift(1).rolling(steps_30d, min_periods=1).mean())
    )
    df["roll_30d_std"] = (
        px["discharge"]
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

def train_model(client: Client, dtrain: xgb.dask.DaskQuantileDMatrix, dtest: xgb.dask.DaskQuantileDMatrix) -> xgb.Booster:
    # TODO: Grid search for hyperparameters

    params = {
        "tree_method":      "hist",
        "device":           "cuda",
        "objective":        "reg:squarederror",
        "eval_metric":      ["rmse", "mae"],
        "max_depth":        6,
        "eta":              0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "seed":             MODEL_SEED,
    }

    # Trains across every GPU worker in the cluster
    output = xgb.dask.train(
        client,
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=25,
    )

    return output["booster"]

def evaluate(client: Client, model: xgb.Booster, test: dd.DataFrame):
    # Predictions are computed on the GPU workers; only summary sums come back
    preds = xgb.dask.inplace_predict(client, model, test[FEATURE_COLS])
    truth = test[TARGET_COL].astype("float64")
    err = truth - preds

    n, sse, sae, sum_y, sum_y2 = dask.compute(
        truth.count(), (err ** 2).sum(), err.abs().sum(), truth.sum(), (truth ** 2).sum()
    )

    rmse = np.sqrt(sse / n)
    mae  = sae / n
    truth_var = sum_y2 - sum_y ** 2 / n
    r2 = 1 - sse / truth_var if truth_var > 0 else float("nan")

    importance = model.get_score(importance_type="gain")
    importance = pd.Series(importance).sort_values(ascending=False)
    print(f"\n── Top 10 features by gain ───────")
    print(importance.head(10).to_string())
    
    # Write metrics to file for easy viewing
    with open("modelresults.md", 'a') as outfile:
        outfile.write("### Model Metrics\n")
        outfile.write(f"RMSE : {rmse:.4f} m/day\n")
        outfile.write(f"MAE  : {mae:.4f} m/day\n")
        outfile.write(f"R²   : {r2:.4f}\n")

    return {"rmse": rmse, "mae": mae, "r2": r2, "feature_importance": importance}

def shap_explainer(client: Client, model: xgb.Booster, df: dd.DataFrame, n_rows: int, non_seasonal_vars: list):
    '''
    Compute SHAP values for every row with GPU TreeSHAP (path-dependent), spread across all GPU workers.
    Preserves datetime for heatmap plotting.
    '''

    print("Calculating SHAP values on GPU workers in shap_explainer...")
    # Output has one column per feature plus a final bias column
    contribs = xgb.dask.predict(client, model, df[FEATURE_COLS], pred_contribs=True)
    contribs = contribs.rename(columns=dict(zip(range(len(FEATURE_COLS) + 1), FEATURE_COLS + ["bias"])))
    contribs["time"] = df["time"]

    # Each worker writes its own partitions to S3 (a directory of parquet files)
    shap_save_path = f'{S3_BUCKET}/cjense/data/testmodel/shap_values_{GLACIER_NAME}_{RESOLUTION}_seed{MODEL_SEED}.parquet'
    contribs.to_parquet(shap_save_path, storage_options=storage_options, write_index=False)

    print(f"SHAP values saved to S3: " + shap_save_path)

    # Plot a random subsample in time order — plotting every row is far too slow
    sample = df.sample(frac=min(1.0, SHAP_PLOT_SAMPLES / n_rows), random_state=MODEL_SEED).compute()
    sample = sample.sort_values("time").reset_index(drop=True)
    combined_time = sample["time"]

    model.set_param({"device": "cuda"})
    sample_contribs = model.predict(xgb.DMatrix(sample[FEATURE_COLS]), pred_contribs=True)
    shap_vals_combined = shap.Explanation(
        values=sample_contribs[:, :-1],
        base_values=sample_contribs[:, -1],
        data=sample[FEATURE_COLS].to_numpy(dtype=np.float32),
        feature_names=FEATURE_COLS,
    )
    ##### Beeswarm plot #####
    shap.plots.beeswarm(shap_vals_combined[:, non_seasonal_vars], show=False, max_display=len(FEATURE_COLS))
    beeswarm_path = f'/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures/{GLACIER_NAME}_beeswarm_{RESOLUTION}_seed{MODEL_SEED}.png'
    # Save figure to .figures/ to upload to GitHub
    plt.savefig('./figures/beeswarm.png', dpi=300)
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(beeswarm_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_beeswarm_{RESOLUTION}_seed{MODEL_SEED}.png')
    plt.clf()

    ##### Heatmap plot #####
    instance_order = np.arange(len(combined_time))

    ax = shap.plots.heatmap(shap_vals_combined[:, non_seasonal_vars], instance_order=instance_order, show=False)
    ax.set_aspect("auto")
    ax.figure.set_size_inches(15, 5)

    # Label the x-axis with the year instead of a raw instance index
    year_change = combined_time.dt.year.ne(combined_time.dt.year.shift(1))
    tick_pos = np.flatnonzero(year_change.to_numpy())
    tick_labels = combined_time.dt.year.iloc[tick_pos].astype(str)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Year")

    # Mark where train data ends and test data begins
    split_idx = np.searchsorted(combined_time.values, np.datetime64(TRAIN_CUTOFF))
    ax.axvline(split_idx - 0.5, color="black", linestyle="--", linewidth=1)

    heatmap_path = f'/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures/{GLACIER_NAME}_heatmap_{RESOLUTION}_seed{MODEL_SEED}.png'
    # Save figure to .figures/ to upload to GitHub
    plt.savefig('./figures/heatmap.png', dpi=300)
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(heatmap_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_heatmap_{RESOLUTION}_seed{MODEL_SEED}.png')
    
    return shap_vals_combined

# def geoshapley_explainer(model: xgb.Booster, train: pd.DataFrame):
#     '''
#     Create a GeoShapley TreeExplainer to explain spatial characteristics of the model.
#     '''
    
#     print("Calculating GeoShapley values in geoshapley_explainer...")
#     tree_explainer = GeoShapleyTreeExplainer(model, g=2)
#     geoshapleyvals = tree_explainer.explain(train)

#     # geoshapleyvals.summary_plot()
    
#     # Save SHAP values to dataframe and CSV
#     csv_save_path = f'{S3_BUCKET}/cjense/data/testmodel/geoshapley_values_{GLACIER_NAME}_{RESOLUTION}_seed{MODEL_SEED}.csv'
#     df_shap = pd.DataFrame(geoshapleyvals.values, columns=FEATURE_COLS)
#     df_shap.to_csv(csv_save_path, index=False)
    
#     print(f"GeoShapley value CSV saved to S3: " + csv_save_path)

#     return geoshapleyvals

def build_flat_cache(cache_path: str):
    '''
    Build the flat feature table from the spatial and non-spatial parquets and write it to S3.
    '''
    # Load spatial and non-spatial variables.
    # This method conserves memory by loading the non-spatial variables lazily and merging once
    spatial_df = pd.read_parquet('s3://gaia/cjense/data/testmodel/velocity_melt_2000_2008.parquet', storage_options=storage_options)
    non_spatial = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet", storage_options=storage_options)
    
    spatial_df = spatial_df.reset_index()
    non_spatial = non_spatial.reset_index()
    non_spatial["time"] = pd.to_datetime(non_spatial["time"]).dt.normalize()
    spatial_df["time"] = pd.to_datetime(spatial_df["time"])

    df = spatial_df.merge(non_spatial, on="time", how="outer")

    # df = df.resample("ME", on='time').mean().reset_index()
    resolution_days = int(RESOLUTION.replace("D", ""))
    df = engineer_features(df, resolution_days=resolution_days)
    
    # TODO: Optimize datatypes (convert to float32) to conserve memory

    print(f"Writing flat parquet to {cache_path} ...")
    df.to_parquet(cache_path, storage_options=storage_options, index=False)
    
    # Record making cached file
    with open("modelresults.md", 'a') as outfile:
        outfile.write("### Data\n")
        outfile.write(f"No cached file found. Made new file at {RESOLUTION} resolution and uploaded it to s3: {cache_path}.")
        df.head().to_markdown(buf=outfile)

def main(client: Client):
    n_workers = len(client.scheduler_info()["workers"])
    print(f"Dask cluster running with {n_workers} GPU worker(s)")

    # Look for existing file of the correct resolution
    cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_{RESOLUTION}.parquet"
    print(f"Looking for cached flat parquet at {cache_path} ...")
    cache_found = fs.exists(cache_path)
    if cache_found:
        print("Found cached file.")
    else:
        print("Did not find cached file. Making new dataframe.")
        build_flat_cache(cache_path)

    # Read the flat table in parallel across the workers
    df = dd.read_parquet(cache_path, columns=FEATURE_COLS + [TARGET_COL, "time"], storage_options=storage_options)
    if df.npartitions < 4 * n_workers:
        df = df.repartition(npartitions=4 * n_workers)

    # Drop NaNs from target feature
    # You can't predict NaN values!
    df = df.dropna(subset=[TARGET_COL]).persist()
    n_rows = len(df)

    if cache_found:
        # Record data size and location
        with open("modelresults.md", 'a') as outfile:
            outfile.write("### Data\n")
            outfile.write(f"Loaded from cache at {cache_path}.")
            outfile.write(f"Loaded from cache: {n_rows:,} rows with non-NaN {TARGET_COL}, {df.memory_usage(deep=True).sum().compute() / 1e9:.2f} GB\n")
            df.head().to_markdown(buf=outfile)

    # Split data into train and test set (time column stays alongside for SHAP plots)
    train, test = df.random_split([0.8, 0.2], random_state=SPLIT_SEED)
    train, test = dask.persist(train, test)
    
    # TODO: Grid search for hyperparameters
    
    local_model_path = f"/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/models/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}.json"
    s3_model_path = f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}.json"
    try:
        # Try to load an existing model
        print("Trying to load existing model...")
        try:
            # Try grabbing model from S3
            fs.get(s3_model_path, local_model_path)
            model = xgb.Booster()
            model.load_model(local_model_path)
            
            print("Model found in S3 bucket.")
            
        except:
            # Try grabbing model from disk
            model = xgb.Booster()
            model.load_model(local_model_path)
            
            print("Model loaded from disk.")
        
        print(f"Loaded existing model {GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}.")
        
    except:
        # If the model doesn't exist yet, train a new model
        print("Existing model not found. Training new model...")

        # Construct distributed DMatrices for training and testing (datetime excluded)
        dtrain = xgb.dask.DaskQuantileDMatrix(client, train[FEATURE_COLS], train[TARGET_COL])
        dtest = xgb.dask.DaskQuantileDMatrix(client, test[FEATURE_COLS], test[TARGET_COL], ref=dtrain)

        model = train_model(client, dtrain, dtest)
    
        # Save the model to disk
        model.save_model(local_model_path)
        print(f"Model saved to disk: " + local_model_path)
        
        # Save the model to S3
        fs.put(local_model_path, s3_model_path)
        print(f"Model saved to S3: " + s3_model_path)
    
    # Calculate model metrics for both loaded and newly trained models
    model.set_param({"device": "cuda"})
    metrics = evaluate(client, model, test)
    
    # Calculate SHAP values on all (train + test) rows
    print("Calculating SHAP values...")
    shapvals = shap_explainer(client, model, df, n_rows, NON_SEASONAL_VARS)
    
    # Calculate GeoShapley values
    # print("Calculating GeoShapley values...")
    # geoshapleyvals = geoshapley_explainer(model, X_train)
    
    return model, metrics, shapvals#, geoshapleyvals

if __name__ == "__main__":
    # One Dask worker per visible GPU (all GPUs allocated to the job unless CUDA_VISIBLE_DEVICES is set)
    with LocalCUDACluster() as cluster, Client(cluster) as client:
        model, metrics, shapvals = main(client)
