"""
Sequential feature selection plus hyperparameter tuning over every non-spatial feature.

Alternates two steps over the whole period, both scored with blocked cross-validation
(blocked_cv.py: each fold holds out one CV_BLOCK_YEARS chunk and trains on the years before
AND after it, with a CV_GAP_MONTHS buffer):
  1. Tune XGBoost hyperparameters with hyperopt (TPE) on the current feature set.
  2. Run scikit-learn's SequentialFeatureSelector with those hyperparameters.
Starting from all candidates, repeat until the selected features stop changing
(or TUNE_ROUNDS is reached), then do a final tuning pass on the selected features.

Every year is used both to choose features/params and to score them, so the CV RMSE
reported here (and by run_xgboost.py) is somewhat optimistic.

main() returns (best_features, best_params):
    best_features : np.ndarray of feature names
    best_params   : dict of XGBRegressor keyword arguments

Usage:
    PERIOD=full pixi run -e local python -u select_features.py
    SFS_DIRECTION=forward HYPEROPT_EVALS=100 pixi run -e local python -u select_features.py
"""
import os
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

from config import *
from preprocess import build_nonspatial, detrend
from blocked_cv import blocked_splits

########## Candidate features ##########
BASE_VARS = [
    "Air Temperature (C)", "Mélange Area (km^2)",
    "Ocean Thermal Forcing (C)", "Terminus Area (km^2)",
    "Basin-Averaged Meltwater Runoff (mmWE/day)",
    "Basin-Averaged Velocity (m/yr)",
    "Mélange Velocity (m/yr)",
    "Near-Terminus Velocity (m/yr)",
    "Near-Terminus Meltwater Runoff (mmWE/day)",
]
TIME_VARS = []#["season_sin", "season_cos", "year_norm"]   # time_days is the same split points as year_norm for trees
LAGS      = ("lag_1step", "lag_30d", "lag_60d", "lag_90d")
LAG_VARS  = [f"{lag}_{var}" for var in BASE_VARS for lag in LAGS]   # no discharge lags (target)

CANDIDATES = BASE_VARS + TIME_VARS + LAG_VARS

########## Selector settings ##########
DIRECTION  = os.environ.get("SFS_DIRECTION", "backward")   # "forward" adds features, "backward" removes them
TOL       = 0.0                                          # stop once a step no longer lowers CV RMSE

########## Hyperparameter tuning settings ##########
HYPEROPT_EVALS = int(os.environ.get("HYPEROPT_EVALS", 50))   # hyperopt trials per tuning pass
TUNE_ROUNDS    = int(os.environ.get("TUNE_ROUNDS", 2))       # max tune -> select rounds before the final tune

# Settings that are never tuned
FIXED_PARAMS = dict(tree_method="hist", device=DEVICE, objective="reg:squarederror", random_state=MODEL_SEED)

# Starting point is DEFAULT_PARAMS from config.py. There is no validation set inside each
# CV fold for early stopping, so the number of rounds is fixed (and tuned) instead.

# Search space for hyperopt
SEARCH_SPACE = {
    "n_estimators":     scope.int(hp.quniform("n_estimators", 100, 1000, 50)),
    "learning_rate":    hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "max_depth":        scope.int(hp.quniform("max_depth", 2, 8, 1)),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1), np.log(50)),
    "subsample":        hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "reg_lambda":       hp.loguniform("reg_lambda", np.log(1e-3), np.log(10)),
    "gamma":            hp.uniform("gamma", 0.0, 1.0),
}


def cv_rmse(model, X, y, cv) -> float:
    return -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()


def make_model(params: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**FIXED_PARAMS, **params)


def tune_params(X: pd.DataFrame, y: np.ndarray, cv) -> tuple[dict, float]:
    """Find the hyperparameters with the lowest CV RMSE on these features (hyperopt TPE)."""
    def objective(params):
        return {"loss": cv_rmse(make_model(params), X, y, cv), "status": STATUS_OK, "params": params}

    trials = Trials()
    fmin(objective, SEARCH_SPACE, algo=tpe.suggest, max_evals=HYPEROPT_EVALS, trials=trials,
         rstate=np.random.default_rng(MODEL_SEED), show_progressbar=False)
    best = trials.best_trial["result"]
    return best["params"], best["loss"]


def select(X: pd.DataFrame, y: np.ndarray, cv, params: dict) -> list:
    """Sequential feature selection with a fixed set of hyperparameters."""
    sfs = SequentialFeatureSelector(
        make_model(params), n_features_to_select="auto", tol=TOL, direction=DIRECTION,
        scoring="neg_root_mean_squared_error", cv=cv,
    )
    sfs.fit(X, y)
    return list(sfs.get_feature_names_out())


def main():
    df = build_nonspatial(feature_cols=CANDIDATES).dropna(subset=[TARGET_COL])

    # Select on the same (detrended) data the model is trained on
    df = df[df["time"].between(PERIOD_START, PERIOD_END)].sort_values("time").reset_index(drop=True)
    if DETREND:
        detrend_cols = [c for c in [*CANDIDATES, TARGET_COL] if c not in DETREND_EXCLUDE]
        df, _ = detrend(df, detrend_cols, np.ones(len(df), dtype=bool))

    # Whole period; names kept as X_train/y_train since every fold trains on most of it
    X_train, y_train = df[CANDIDATES], df[TARGET_COL].to_numpy()
    cv = blocked_splits(df["time"])
    print(f"Period {PERIOD}: {len(df)} timesteps, {len(cv)} CV chunks of {CV_BLOCK_YEARS} yr "
          f"(gap {CV_GAP_MONTHS} mo), {len(CANDIDATES)} candidate features")

    # Alternate tuning and selection until the feature set stops changing
    features = list(CANDIDATES)
    history = []
    for round_ in range(1, TUNE_ROUNDS + 1):
        t0 = time.time()
        params, loss = tune_params(X_train[features], y_train, cv)
        print(f"[round {round_}] tuned on {len(features)} features: CV RMSE {loss:.4f} ({time.time() - t0:.0f} s)")

        t0 = time.time()
        selected = select(X_train[features], y_train, cv, params)
        print(f"[round {round_}] {DIRECTION} selection kept {len(selected)} features ({time.time() - t0:.0f} s): {selected}")
        history.append({"round": round_, "tuned on": len(features), "tuning CV RMSE": loss, "selected": len(selected)})

        if set(selected) == set(features):
            break
        features = selected

    # Final tuning on the selected features
    t0 = time.time()
    best_params, best_loss = tune_params(X_train[selected], y_train, cv)
    print(f"[final] tuned on {len(selected)} selected features: CV RMSE {best_loss:.4f} ({time.time() - t0:.0f} s)")
    best_features = np.array(selected)
    print(f"Best features: {best_features.tolist()}")
    print(f"Best params:   {best_params}")

    # Compare against the untuned defaults and the current FEATURE_COLS
    subsets = {
        "selected, tuned params":           (selected, best_params),
        "selected, default params":         (selected, DEFAULT_PARAMS),
        "manual FEATURE_COLS, default":     (MANUAL_FEATURE_COLS, DEFAULT_PARAMS),
        "all candidates, default":          (CANDIDATES, DEFAULT_PARAMS),
    }
    rows = []
    for name, (cols, params) in subsets.items():
        # FEATURE_COLS may be empty or hold columns that aren't candidates (e.g. spatial ones)
        cols = [c for c in cols if c in CANDIDATES]
        if not cols:
            print(f"Skipping comparison '{name}': no candidate features in it.")
            continue
        model = make_model(params)
        rows.append({
            "subset": name,
            "n_features": len(cols),
            "Blocked CV RMSE": cv_rmse(model, X_train[cols], y_train, cv),
        })
    summary = pd.DataFrame(rows).set_index("subset")
    print(summary.round(4).to_string())

    out_path = f"featureselection_{PERIOD}.md"
    with open(out_path, "w") as f:
        f.write(f"## Feature selection + tuning: {PERIOD} ({DIRECTION}, {PERIOD_START} to {PERIOD_END})\n\n")
        f.write(f"{len(CANDIDATES)} candidates, blocked CV ({len(cv)} chunks of {CV_BLOCK_YEARS} yr, "
                f"{CV_GAP_MONTHS}-month gap), {HYPEROPT_EVALS} hyperopt trials per tuning pass.\n\n")
        f.write("### Selected features\n")
        f.writelines(f"- {c}\n" for c in selected)
        f.write("\n### Tuned parameters\n")
        f.writelines(f"- {k}: {v:.4g}\n" if isinstance(v, float) else f"- {k}: {v}\n" for k, v in best_params.items())
        f.write("\n### Rounds\n")
        f.write(pd.DataFrame(history).round(4).to_markdown(index=False) + "\n\n")
        f.write("### Comparison\n")
        f.write(summary.round(4).to_markdown() + "\n\n")
        f.write("CV RMSE is the mean over chunks. Features and parameters were chosen on these same chunks, "
                "so it is somewhat optimistic.\n")
    print(f"Results written to {out_path}")

    # Save the choice for the pipeline (config.py loads it when USE_FEATURE_SELECTION is True).
    # hyperopt returns numpy scalars, so convert to plain Python for JSON.
    selection = {
        "features": selected,
        "params": {k: (int(v) if k in ("n_estimators", "max_depth") else float(v)) for k, v in best_params.items()},
        "cv_rmse": float(best_loss),
        "cv": f"blocked, {len(cv)} chunks of {CV_BLOCK_YEARS} yr, {CV_GAP_MONTHS}-month gap",
        "direction": DIRECTION,
        "hyperopt_evals": HYPEROPT_EVALS,
        "candidates": CANDIDATES,
        "detrended": DETREND,
        "period": [PERIOD_START, PERIOD_END],
    }
    with fs.open(SELECTION_PATH, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"Selection saved to S3: {SELECTION_PATH}")

    return best_features, best_params


if __name__ == "__main__":
    best_features, best_params = main()
