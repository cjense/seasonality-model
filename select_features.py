"""
Sequential feature selection over every non-spatial feature.

Runs scikit-learn's SequentialFeatureSelector with an XGBoost regressor on the TRAINING
period only, scoring each candidate subset with time-ordered cross-validation
(TimeSeriesSplit: always train on earlier timesteps, validate on later ones).
The held-out test period is never used to choose features; it is only reported
at the end for the selected subset and the current FEATURE_COLS, for reference.

Usage:
    PERIOD=pre2015 pixi run -e local python -u select_features.py
    SFS_DIRECTION=backward PERIOD=post2015 pixi run -e local python -u select_features.py
"""
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from config import *
from preprocess import build_nonspatial

########## Candidate features ##########
BASE_VARS = [
    "Air Temperature (C)", "Mélange Area (km^2)",
    "Ocean Thermal Forcing (C)", "Terminus Area (km^2)",
    "Basin-Averaged Meltwater Runoff (mmWE/day)",
]
TIME_VARS = ["season_sin", "season_cos", "year_norm"]   # time_days is the same split points as year_norm for trees
LAGS      = ("lag_1step", "lag_30d", "lag_60d", "lag_90d")
LAG_VARS  = [f"{lag}_{var}" for var in BASE_VARS for lag in LAGS]   # no discharge lags (target)

CANDIDATES = BASE_VARS + TIME_VARS + LAG_VARS

########## Selector settings ##########
DIRECTION  = os.environ.get("SFS_DIRECTION", "backward")   # "forward" adds features, "backward" removes them
N_SPLITS   = 5                                            # TimeSeriesSplit folds within the training period
TOL        = 0.0                                          # stop once a step no longer lowers CV RMSE

# Same tree settings as run_xgboost.py. There is no validation set inside each CV fold for
# early stopping, so use a fixed number of rounds instead.
XGB_PARAMS = dict(
    tree_method="hist", device=DEVICE, objective="reg:squarederror",
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    random_state=MODEL_SEED,
)


def cv_rmse(model, X, y, cv) -> float:
    return -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()


def test_rmse(model, X_train, y_train, X_test, y_test) -> float:
    model.fit(X_train, y_train)
    return float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))


def main():
    df = build_nonspatial(feature_cols=CANDIDATES).dropna(subset=[TARGET_COL])
    train = df[df["time"].between(PERIOD_START, TRAIN_CUTOFF, inclusive="left")].sort_values("time")
    test  = df[df["time"].between(TEST_START, PERIOD_END)].sort_values("time")
    X_train, y_train = train[CANDIDATES], train[TARGET_COL].to_numpy()
    X_test,  y_test  = test[CANDIDATES],  test[TARGET_COL].to_numpy()
    print(f"Period {PERIOD}: {len(train)} train / {len(test)} test timesteps, {len(CANDIDATES)} candidate features")

    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    model = xgb.XGBRegressor(**XGB_PARAMS)

    print(f"Running {DIRECTION} sequential feature selection ({N_SPLITS}-fold time-series CV)...")
    t0 = time.time()
    sfs = SequentialFeatureSelector(
        model, n_features_to_select="auto", tol=TOL, direction=DIRECTION,
        scoring="neg_root_mean_squared_error", cv=cv,
    )
    sfs.fit(X_train, y_train)
    selected = list(sfs.get_feature_names_out())
    print(f"Done in {time.time() - t0:.0f} s. Selected {len(selected)} features: {selected}")

    # Compare the selected subset with the current FEATURE_COLS and with every candidate
    subsets = {"selected": selected, "current FEATURE_COLS": FEATURE_COLS, "all candidates": CANDIDATES}
    rows = []
    for name, cols in subsets.items():
        rows.append({
            "subset": name,
            "n_features": len(cols),
            "CV RMSE (train period)": cv_rmse(model, X_train[cols], y_train, cv),
            "Test RMSE (reference only)": test_rmse(model, X_train[cols], y_train, X_test[cols], y_test),
        })
    summary = pd.DataFrame(rows).set_index("subset")
    print(summary.round(4).to_string())

    out_path = f"featureselection_{PERIOD}.md"
    with open(out_path, "w") as f:
        f.write(f"## Feature selection: {PERIOD} ({DIRECTION}, train {PERIOD_START} to {TRAIN_CUTOFF})\n\n")
        f.write(f"{len(CANDIDATES)} candidates, {N_SPLITS}-fold TimeSeriesSplit CV on the training period, "
                f"XGBoost {XGB_PARAMS['n_estimators']} rounds.\n\n")
        f.write("### Selected features\n")
        f.writelines(f"- {c}\n" for c in selected)
        f.write("\n### Comparison\n")
        f.write(summary.round(4).to_markdown() + "\n\n")
        f.write("Test RMSE is shown for reference only; features were chosen on CV RMSE.\n")
    print(f"Results written to {out_path}")

    return selected, summary


if __name__ == "__main__":
    selected, summary = main()
