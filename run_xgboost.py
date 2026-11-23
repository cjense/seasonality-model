import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import skforecast.metrics as skmetrics

from config import *
from blocked_cv import blocked_splits, chunk_names


def booster_params(params: dict) -> tuple[dict, int]:
    """
    Convert MODEL_PARAMS (XGBRegressor names, from select_features.py or DEFAULT_PARAMS)
    to xgb.train arguments. learning_rate, reg_lambda, gamma etc. are valid native aliases.
    """
    params = dict(params)
    n_rounds = params.pop("n_estimators")
    return {
        "tree_method": "hist",
        "device":      DEVICE,
        "objective":   "reg:squarederror",
        "eval_metric": ["rmse", "mae", "mape"],
        "seed":        MODEL_SEED,
        **params,
    }, n_rounds

def train_model(df: pd.DataFrame, verbose: bool = False) -> xgb.Booster:
    # Number of trees comes from tuning (blocked CV); no early stopping, which would need a validation set
    params, n_rounds = booster_params(MODEL_PARAMS)
    dtrain = xgb.QuantileDMatrix(df[FEATURE_COLS], label=df[TARGET_COL])
    return xgb.train(params, dtrain, num_boost_round=n_rounds,
                     evals=[(dtrain, "train")] if verbose else (), verbose_eval=25 if verbose else False)

def cross_validate(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Blocked CV over the whole period with the final features and params (folds from blocked_cv.py):
    each chunk is predicted by a model trained on all other chunks, before and after it.
    Returns the out-of-fold prediction for every row and a per-chunk error table.
    """
    oof = pd.Series(np.nan, index=data.index, name="pred")
    rows = []
    for name, (fit_pos, val_pos) in zip(chunk_names(data["time"]), blocked_splits(data["time"])):
        fit, val = data.iloc[fit_pos], data.iloc[val_pos]
        model = train_model(fit)
        oof.iloc[val_pos] = model.predict(xgb.DMatrix(val[FEATURE_COLS]))
        err = oof.iloc[val_pos].to_numpy() - val[TARGET_COL].to_numpy()
        rows.append({
            "held out": name,
            "timesteps": val["time"].nunique(),
            "RMSE":     float(np.sqrt(np.mean(err ** 2))),
            "MAE":      float(np.mean(np.abs(err))),
        })
    per_chunk = pd.DataFrame(rows).set_index("held out")
    print(per_chunk.round(4).to_string())
    return oof, per_chunk

def evaluate(oof: pd.Series, data: pd.DataFrame, per_chunk: pd.DataFrame, model: xgb.Booster):
    """Metrics on the pooled out-of-fold predictions: every timestep predicted by a model that never saw its chunk."""
    preds = oof.to_numpy()
    truth = data[TARGET_COL].to_numpy()

    rmse = np.sqrt(np.mean((preds - truth) ** 2))
    mae  = np.mean(np.abs(preds - truth))
    truth_var = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - np.sum((truth - preds) ** 2) / truth_var if truth_var > 0 else float("nan")

    # Scaled errors: relative to a naive one-step forecast of the target series
    y_scale = data.groupby("time")[TARGET_COL].mean().to_numpy()
    rmsse = skmetrics.root_mean_squared_scaled_error(truth, preds, y_scale)
    mase = skmetrics.mean_absolute_scaled_error(truth, preds, y_scale)
    smape = skmetrics.symmetric_mean_absolute_percentage_error(truth, preds)

    # TODO: Calculate seasonal (phase, amplitude, etc.) errors

    # Importance from the final model, trained on all chunks
    importance = model.get_score(importance_type="gain")
    importance = pd.Series(importance).sort_values(ascending=False)

    # Write metrics to file for easy viewing
    with open(RESULTS_PATH, 'a') as outfile:
        outfile.write(NOTES + "\n\n")
        outfile.write(f"### Model Metrics (blocked CV, {len(per_chunk)} chunks of {CV_BLOCK_YEARS} yr, "
                      f"{CV_GAP_MONTHS}-month gap; pooled out-of-fold predictions)\n")
        outfile.write(f"- RMSE   : {rmse:.4f} Gt/yr\n")
        outfile.write(f"- MAE    : {mae:.4f} Gt/yr\n")
        outfile.write(f"- R²     : {r2:.4f}\n")
        outfile.write(f"- RMSSE  : {rmsse:.4f}\n")
        outfile.write(f"- MASE   : {mase:.4f}\n")
        outfile.write(f"- SMAPE  : {smape:.4f}\n")
        if USE_FEATURE_SELECTION:
            outfile.write("\nFeatures and params were selected with the same chunks, so these are somewhat optimistic.\n")
        outfile.write("\n### Error by held-out chunk\n")
        outfile.write(per_chunk.round(4).to_markdown() + "\n\n")
        outfile.write(f"Mean of per-chunk RMSE: {per_chunk['RMSE'].mean():.4f} Gt/yr "
                      f"(range {per_chunk['RMSE'].min():.4f} to {per_chunk['RMSE'].max():.4f})\n\n")
        outfile.write(f"### Top 10 features by gain (final model, all data)\n")
        outfile.write(importance.head(10).to_markdown() + "\n\n")

    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsse": rmsse, "mase": mase, "smape": smape,
            "per_chunk": per_chunk, "feature_importance": importance}

def write_model_setup():
    """Record which features and hyperparameters were used, and where they came from."""
    with open(RESULTS_PATH, 'a') as outfile:
        outfile.write("### Features and hyperparameters\n")
        if SELECTION is not None:
            outfile.write(f"From select_features.py ({SELECTION['direction']} selection, {SELECTION['hyperopt_evals']} hyperopt "
                          f"trials, {SELECTION.get('cv', 'CV')}, CV RMSE {SELECTION['cv_rmse']:.4f}): {SELECTION_PATH}\n\n")
        else:
            outfile.write("Hand-picked FEATURE_COLS and DEFAULT_PARAMS from config.py (USE_FEATURE_SELECTION is False).\n\n")
        outfile.writelines(f"- {c}\n" for c in FEATURE_COLS)
        outfile.write("\n")
        outfile.writelines(f"- {k}: {v:.4g}\n" if isinstance(v, float) else f"- {k}: {v}\n" for k, v in MODEL_PARAMS.items())
        outfile.write("\n")

def plot_predictions(oof: pd.Series, data: pd.DataFrame):
    """Truth vs. out-of-fold predictions; alternating shading marks the held-out chunks."""
    # Many pixels per timestep in spatial runs, so average truth and predictions over space
    ts = data[["time", TARGET_COL]].assign(pred=oof).groupby("time")[[TARGET_COL, "pred"]].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (_, val_pos) in enumerate(blocked_splits(data["time"])):
        if i % 2:
            times = data["time"].iloc[val_pos]
            ax.axvspan(times.min(), times.max() + pd.offsets.MonthEnd(0), color="gray", alpha=0.1, linewidth=0)
    ax.plot(ts.index, ts[TARGET_COL], color="black", linewidth=1, label="truth")
    ax.plot(ts.index, ts["pred"], color="tab:blue", linewidth=1, linestyle="--", label="prediction (held-out chunk)")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Detrended {TARGET_COL}" if DETREND else TARGET_COL)
    ax.set_title(f"Blocked CV {PERIOD_START} to {PERIOD_END}: each {CV_BLOCK_YEARS}-yr chunk predicted by a model trained on the others")
    ax.legend()
    xgboost_fig_path = f'{FIGURE_DIR}/{GLACIER_NAME}_xgbresult_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.png'
    # Save figure to .figures/ to upload to GitHub
    plt.savefig(f'./figures/xgboost_result_{PERIOD}.png', dpi=300, bbox_inches='tight')
    plt.savefig(xgboost_fig_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(xgboost_fig_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_xgbresult_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.png')
    plt.close(fig)

def main():
    require_selection()

    # Load the dataset written by preprocess.py (whole period, sorted by time)
    data = pd.read_parquet(DATA_PATH, storage_options=storage_options).dropna(subset=[TARGET_COL]).reset_index(drop=True)

    print("Cross-validating over held-out chunks...")
    oof, per_chunk = cross_validate(data)

    # Final model on all data: this is the one saved and explained by SHAP
    print("Training final model on all data...")
    model = train_model(data, verbose=True)

    write_model_setup()
    metrics = evaluate(oof, data, per_chunk, model)
    plot_predictions(oof, data)

    # Save the model to disk
    model.save_model(LOCAL_MODEL_PATH)
    print(f"Model saved to disk: " + LOCAL_MODEL_PATH)

    # Save the model to S3
    fs.put(LOCAL_MODEL_PATH, S3_MODEL_PATH)
    print(f"Model saved to S3: " + S3_MODEL_PATH)

    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
