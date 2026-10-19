import numpy as np
import pandas as pd
import xgboost as xgb
import s3fs
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import *

def shap_explainer(model: xgb.Booster, train: pd.DataFrame, test: pd.DataFrame, time_train: pd.Series, time_test: pd.Series, features: list):
    '''
    Compute path-dependent TreeSHAP values on the GPU (XGBoost pred_contribs) to quantify input variable influence on output variables.
    Preserves datetime for heatmap plotting.
    '''

    # Combine training and testing data (features only, datetime separate)
    combined_features = pd.concat([train, test], ignore_index=True)
    combined_time = pd.concat([time_train, time_test], ignore_index=True)

    # Sort by time and get the sort indices
    sort_idx = combined_time.argsort()
    combined_features = combined_features.iloc[sort_idx].reset_index(drop=True)
    combined_time = combined_time.iloc[sort_idx].reset_index(drop=True)

    # Compute SHAP values on the GPU with XGBoost's built-in GPUTreeSHAP.
    print("Calculating SHAP values on GPU...")
    model.set_param({"device": "cuda"})
    contribs = model.predict(xgb.DMatrix(combined_features), pred_contribs=True)
    shap_vals_combined = shap.Explanation(
        values=contribs[:, :-1],
        base_values=contribs[:, -1],
        data=combined_features.to_numpy(),
        feature_names=list(combined_features.columns),
    )

    # Save SHAP values to parquet, keyed by (time, x, y) so they can be joined back to the flat feature parquet
    shap_save_path = f'{S3_BUCKET}/cjense/data/testmodel/shap_values_{GLACIER_NAME}_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.parquet'
    df_shap = pd.DataFrame(shap_vals_combined.values, columns=[f"shap_{c}" for c in FEATURE_COLS])
    df_shap.insert(0, "time", combined_time)
    df_shap.insert(1, "x", combined_features["x"])
    df_shap.insert(2, "y", combined_features["y"])
    df_shap["base_value"] = shap_vals_combined.base_values
    df_shap.to_parquet(shap_save_path, storage_options=storage_options, index=False)

    print(f"SHAP value parquet saved to S3: " + shap_save_path)

    ##### Beeswarm plot #####
    # Plot a random subsample -- a matplotlib scatter of every row is extremely slow.
    # The full SHAP values are still saved to parquet above and used for the heatmap below.
    n_rows = len(shap_vals_combined.values)
    rng = np.random.default_rng(MODEL_SEED)
    plot_idx = np.sort(rng.choice(n_rows, size=min(n_rows, BEESWARM_MAX_POINTS), replace=False))
    print(f"Plotting beeswarm with {len(plot_idx):,} of {n_rows:,} rows...")
    shap.plots.beeswarm(shap_vals_combined[plot_idx][:, features], show=False, max_display=len(FEATURE_COLS),
                        plot_size=(14, 8))
    plt.title("SHAP values (path-dependent TreeSHAP, GPU)")
    plt.tight_layout()
    beeswarm_path = f'/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures/{GLACIER_NAME}_beeswarm_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png'
    # Save figure to .figures/ to upload to GitHub
    plt.savefig('./figures/beeswarm.png', dpi=300, bbox_inches='tight')
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(beeswarm_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_beeswarm_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png')
    plt.clf()

    ##### Heatmap plot #####
    # Average SHAP values over all pixels at each timestep -> one heatmap column per timestep
    time_key = combined_time.to_numpy()
    shap_by_time = pd.DataFrame(shap_vals_combined.values, columns=FEATURE_COLS).groupby(time_key).mean()
    data_by_time = combined_features[FEATURE_COLS].groupby(time_key).mean()
    base_by_time = pd.Series(shap_vals_combined.base_values).groupby(time_key).mean()
    times = pd.Series(shap_by_time.index)

    shap_vals_by_time = shap.Explanation(
        values=shap_by_time.to_numpy(),
        base_values=base_by_time.to_numpy(),
        data=data_by_time.to_numpy(),
        feature_names=FEATURE_COLS,
    )
    instance_order = np.arange(len(times))

    ax = shap.plots.heatmap(shap_vals_by_time[:, features], instance_order=instance_order, show=False)
    ax.set_aspect("auto")
    ax.figure.set_size_inches(15, 5)

    # Label the x-axis with the year instead of a raw instance index
    year_change = times.dt.year.ne(times.dt.year.shift(1))
    tick_pos = np.flatnonzero(year_change.to_numpy())
    tick_labels = times.dt.year.iloc[tick_pos].astype(str)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Year")
    ax.set_title("Monthly mean SHAP values over all pixels (path-dependent TreeSHAP, GPU)")

    heatmap_path = f'/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures/{GLACIER_NAME}_heatmap_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png'
    # Save figure to .figures/ to upload to GitHub
    plt.savefig('./figures/heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(heatmap_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_heatmap_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png')
    
    return shap_vals_combined

def main():
    # Load train and test sets written by preprocess.py
    train = pd.read_parquet(TRAIN_PATH, storage_options=storage_options)
    test = pd.read_parquet(TEST_PATH, storage_options=storage_options)

    # Load the model written by run_xgboost.py
    model = xgb.Booster()
    try:
        # Try grabbing model from S3
        fs.get(S3_MODEL_PATH, LOCAL_MODEL_PATH)
        print("Model found in S3 bucket.")
    except Exception:
        # Fall back to the copy on disk
        print("Model not found in S3. Loading from disk.")
    model.load_model(LOCAL_MODEL_PATH)

    # Calculate SHAP values
    shapvals = shap_explainer(model, train[FEATURE_COLS], test[FEATURE_COLS], train["time"], test["time"], FEATURE_COLS)

    return model, shapvals

if __name__ == "__main__":
    model, shapvals = main()
