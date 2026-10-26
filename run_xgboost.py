import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import skforecast.metrics as skmetrics

from config import *


def train_model(dtrain: xgb.QuantileDMatrix, dtest: xgb.QuantileDMatrix) -> xgb.Booster:
    # TODO: Grid search for hyperparameters

    params = {
        "tree_method":      "hist",
        "device":           "cuda",
        "objective":        "reg:squarederror",
        "eval_metric":      ["rmse", "mae", "mape"],
        "max_depth":        6,
        "eta":              0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "seed":             MODEL_SEED,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=25,
    )

    return model

def evaluate(model: xgb.Booster, dtest: xgb.QuantileDMatrix, train: pd.DataFrame):
    preds = model.predict(dtest)
    truth = dtest.get_label()
    # Mask NaN in truth only — can't compute metrics on unknown ground truth

    rmse = np.sqrt(np.mean((preds - truth) ** 2))
    mae  = np.mean(np.abs(preds - truth))
    truth_var = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - np.sum((truth - preds) ** 2) / truth_var if truth_var > 0 else float("nan")
    
    y_train = train[TARGET_COL].dropna().to_numpy()
    rmsse = skmetrics.root_mean_squared_scaled_error(truth, preds, y_train)
    mase = skmetrics.mean_absolute_scaled_error(truth, preds, y_train)
    smape = skmetrics.symmetric_mean_absolute_percentage_error(truth, preds)
    
    # TODO: Calculate seasonal (phase, amplitude, etc.) errors

    importance = model.get_score(importance_type="gain")
    importance = pd.Series(importance).sort_values(ascending=False)
    
    # Write metrics to file for easy viewing
    with open("modelresults.md", 'a') as outfile:
        outfile.write(NOTES + "\n\n")
        outfile.write("### Model Metrics\n")
        outfile.write(f"- R²     : {r2:.4f}\n")
        outfile.write(f"- RMSSE  : {rmsse:.4f}\n")
        outfile.write(f"- MASE   : {mase:.4f}\n")
        outfile.write(f"- SMAPE  : {smape:.4f}\n")
        outfile.write(f"\n### Top 10 features by gain\n")
        outfile.write(importance.head(10).to_markdown() + "\n\n")

    return {"rmse": rmse, "mae": mae, "r2": r2, "feature_importance": importance}

def plot_predictions(model: xgb.Booster, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Time series of truth vs. predictions over both the train and test periods."""
    
    fig, ax = plt.subplots(figsize=(12, 4))
    for df, label, color in [(train_df, "train", "tab:blue"), (test_df, "test", "tab:orange")]:
        # Many pixels per timestep, so average truth and predictions over space
        df = (df.assign(pred=model.predict(xgb.DMatrix(df[FEATURE_COLS])))
                .groupby("time")[[TARGET_COL, "pred"]].mean())
        ax.plot(df.index, df[TARGET_COL], color="black", linewidth=1,
                label="truth" if label == "train" else None)
        ax.plot(df.index, df["pred"], color=color, linewidth=1, linestyle="--",
                label=f"prediction ({label})")

    # mark where train data ends and test data begins
    ax.axvline(pd.Timestamp(TRAIN_CUTOFF), color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    xgboost_fig_path = f'/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures/{GLACIER_NAME}_xgbresult_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png'
    # Save figure to .figures/ to upload to GitHub
    plt.savefig('./figures/xgboost_result.png', dpi=300, bbox_inches='tight')
    plt.savefig(xgboost_fig_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(xgboost_fig_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_xgbresult_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.png')
    plt.close(fig)

def main():
    # Load train and test sets written by preprocess.py
    train = pd.read_parquet(TRAIN_PATH, storage_options=storage_options)
    test = pd.read_parquet(TEST_PATH, storage_options=storage_options)

    # Construct QuantileDMatrices for training and testing (datetime excluded)
    dtrain = xgb.QuantileDMatrix(train[FEATURE_COLS], label=train[TARGET_COL])
    dtest = xgb.QuantileDMatrix(test[FEATURE_COLS], label=test[TARGET_COL], ref=dtrain)

    print("Training new model...")
    model = train_model(dtrain, dtest)

    # Calculate model metrics
    metrics = evaluate(model, dtest, train)
    plot_predictions(model, train, test)

    # Save the model to disk
    model.save_model(LOCAL_MODEL_PATH)
    print(f"Model saved to disk: " + LOCAL_MODEL_PATH)

    # Save the model to S3
    fs.put(LOCAL_MODEL_PATH, S3_MODEL_PATH)
    print(f"Model saved to S3: " + S3_MODEL_PATH)

    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
