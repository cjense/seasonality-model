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


def geoshapley_explainer(model: xgb.Booster, train: pd.DataFrame):
    '''
    Create a GeoShapley TreeExplainer to explain spatial characteristics of the model.
    '''
    
    print("Calculating GeoShapley values...")
    tree_explainer = GeoShapleyTreeExplainer(model, g=2)
    geoshapleyvals = tree_explainer.explain(train)

    # geoshapleyvals.summary_plot()
    
    # Save SHAP values to dataframe and CSV
    csv_save_path = f'{S3_BUCKET}/cjense/data/testmodel/geoshapley_values_{GLACIER_NAME}_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.csv'
    df_shap = pd.DataFrame(geoshapleyvals.values, columns=FEATURE_COLS)
    df_shap.to_csv(csv_save_path, index=False, storage_options=storage_options)
    
    print(f"GeoShapley value CSV saved to S3: " + csv_save_path)

    return geoshapleyvals

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

    # Calculate GeoShapley values
    # TODO
    # geoshapleyvals = geoshapley_explainer(model, train[FEATURE_COLS])

    # return model, geoshapleyvals

if __name__ == "__main__":
    # model, geoshapleyvals = main()
    continue
