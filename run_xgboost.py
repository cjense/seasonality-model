import numpy as np
import pandas as pd
import xgboost as xgb
import s3fs
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from geoshapley import GeoShapleyTreeExplainer

from config import *


def train_model(dtrain: xgb.QuantileDMatrix, dtest: xgb.QuantileDMatrix) -> xgb.Booster:
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

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=25,
    )

    return model

def evaluate(model: xgb.Booster, dtest: xgb.QuantileDMatrix):
    preds = model.predict(dtest)
    truth = dtest.get_label()

    # Mask NaN in truth only — can't compute metrics on unknown ground truth
    valid = ~np.isnan(truth)
    preds = preds[valid]
    truth = truth[valid]

    rmse = np.sqrt(np.mean((preds - truth) ** 2))
    mae  = np.mean(np.abs(preds - truth))
    truth_var = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - np.sum((truth - preds) ** 2) / truth_var if truth_var > 0 else float("nan")

    importance = model.get_score(importance_type="gain")
    importance = pd.Series(importance).sort_values(ascending=False)
    
    # Write metrics to file for easy viewing
    with open("modelresults.md", 'a') as outfile:
        outfile.write(NOTES + "\n")
        outfile.write("### Model Metrics\n")
        outfile.write(f"RMSE : {rmse:.4f} Gt/yr\n")
        outfile.write(f"MAE  : {mae:.4f} Gt/yr\n")
        outfile.write(f"R²   : {r2:.4f}\n")
        outfile.write(f"\n── Top 10 features by gain ───────")
        outfile.write(importance.head(10).to_string())

    return {"rmse": rmse, "mae": mae, "r2": r2, "feature_importance": importance}

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
    metrics = evaluate(model, dtest)

    # Save the model to disk
    model.save_model(LOCAL_MODEL_PATH)
    print(f"Model saved to disk: " + LOCAL_MODEL_PATH)

    # Save the model to S3
    fs.put(LOCAL_MODEL_PATH, S3_MODEL_PATH)
    print(f"Model saved to S3: " + S3_MODEL_PATH)

    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
