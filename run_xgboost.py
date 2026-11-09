import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import skforecast.metrics as skmetrics
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

from config import *


def n_visible_gpus() -> int:
    '''Number of GPUs this job can use: N_GPUS override, else CUDA_VISIBLE_DEVICES (set by Slurm), else nvidia-smi.'''
    if os.environ.get("N_GPUS"):
        return int(os.environ["N_GPUS"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([d for d in visible.split(",") if d.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(line.startswith("GPU ") for line in out.splitlines()))
    except Exception:
        return 1


def time_block_folds(times: pd.Series, nfold: int, gap_days: int) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Split rows into `nfold` contiguous blocks of timesteps. Each block is the validation set once;
    training rows within `gap_days` of the block are dropped so lag features / autocorrelation
    can't leak validation-period information into training.
    '''
    times = times.reset_index(drop=True)
    gap = pd.Timedelta(days=gap_days)
    folds = []
    for block in np.array_split(np.sort(times.unique()), nfold):
        start, end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        val = times.between(start, end).to_numpy()
        trn = ((times < start - gap) | (times > end + gap)).to_numpy()
        folds.append((np.flatnonzero(trn), np.flatnonzero(val)))
    return folds


def train_model(train: pd.DataFrame, dtrain: xgb.QuantileDMatrix, dtest: xgb.QuantileDMatrix) -> xgb.Booster:
    '''
    Tune tree hyperparameters with hyperopt on a subset of training timesteps using time-blocked CV
    (folds trained in parallel, one GPU each), pick the number of rounds by early stopping on the end of the
    training period, then refit on the full `dtrain` with a lower learning rate. The test set is never used to fit.
    '''
    # 1. Subsample whole timesteps (all pixels at a timestep are highly correlated, so this loses little)
    rng = np.random.default_rng(MODEL_SEED)
    all_times = train["time"].unique()
    n_keep = max(TUNE_NFOLD * 2, int(round(len(all_times) * TUNE_TIME_FRAC)))
    keep = rng.choice(all_times, size=min(n_keep, len(all_times)), replace=False)
    tune_df = train[train["time"].isin(keep)].reset_index(drop=True)

    # 4. Time-blocked folds, built once and reused by every trial
    folds = time_block_folds(tune_df["time"], TUNE_NFOLD, TUNE_GAP_DAYS)
    X, y = tune_df[FEATURE_COLS], tune_df[TARGET_COL]
    fold_data = []
    for trn_idx, val_idx in folds:
        dtr = xgb.QuantileDMatrix(X.iloc[trn_idx], label=y.iloc[trn_idx])
        dval = xgb.QuantileDMatrix(X.iloc[val_idx], label=y.iloc[val_idx], ref=dtr)
        fold_data.append((dtr, dval))

    # One worker thread per GPU; each runs its assigned folds in turn
    n_gpus = min(n_visible_gpus(), TUNE_NFOLD)
    gpu_folds = [[k for k in range(TUNE_NFOLD) if k % n_gpus == g] for g in range(n_gpus)]
    print(f"Tuning on {len(keep)}/{len(all_times)} timesteps ({len(tune_df):,} rows), "
          f"{TUNE_NFOLD} time-block folds on {n_gpus} GPU(s), {TUNE_MAX_EVALS} trials")

    # The target is identical for every pixel at a timestep, so the number of independent samples is
    # the number of timesteps, not rows. min_child_weight, reg_lambda and gamma all scale with row
    # count (hessian = 1 per row for squared error), so they're tuned in units of "timesteps' worth
    # of pixels" and multiplied by pixels-per-timestep. E.g. min_child_timesteps=10 forces every leaf
    # to contain at least ~10 timesteps of data, so a leaf can't memorize a single timestep.
    pixels_per_timestep = len(train) / len(all_times)

    space = {
        'max_depth':           hp.quniform('max_depth', 2, 6, 1),
        'subsample':           hp.uniform('subsample', 0.5, 1),
        'colsample_bytree':    hp.uniform('colsample_bytree', 0.3, 1),
        'min_child_timesteps': hp.loguniform('min_child_timesteps', np.log(1), np.log(100)),
        'lambda_timesteps':    hp.loguniform('lambda_timesteps', np.log(0.01), np.log(100)),
        'gamma_timesteps':     hp.loguniform('gamma_timesteps', np.log(1e-3), np.log(10)),
    }

    def tree_params(hyperparams, eta):
        return {
            "tree_method":      "hist",
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "eta":              eta,
            "max_depth":        int(hyperparams['max_depth']),
            "subsample":        hyperparams['subsample'],
            "colsample_bytree": hyperparams['colsample_bytree'],
            "min_child_weight": hyperparams['min_child_timesteps'] * pixels_per_timestep,
            "reg_lambda":       hyperparams['lambda_timesteps'] * pixels_per_timestep,
            "gamma":            hyperparams['gamma_timesteps'] * pixels_per_timestep,
            "seed":             MODEL_SEED,
        }

    def run_folds_on_gpu(params, gpu, fold_ids):
        results = []
        for k in fold_ids:
            dtr, dval = fold_data[k]
            bst = xgb.train(
                {**params, "device": f"cuda:{gpu}"},
                dtr,
                num_boost_round=TUNE_MAX_ROUNDS,
                evals=[(dval, "val")],
                early_stopping_rounds=TUNE_EARLY_STOP,
                verbose_eval=False,
            )
            results.append((bst.best_score, bst.best_iteration))
        return results

    def objective(hyperparams):
        params = tree_params(hyperparams, TUNE_ETA)
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = [pool.submit(run_folds_on_gpu, params, g, ids) for g, ids in enumerate(gpu_folds)]
            results = [r for f in futures for r in f.result()]
        rmses, iters = zip(*results)
        return {'loss': float(np.mean(rmses)), 'status': STATUS_OK, 'best_iterations': list(iters)}

    t0 = time.time()
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=TUNE_MAX_EVALS, trials=trials,
                rstate=np.random.default_rng(MODEL_SEED))
    tune_minutes = (time.time() - t0) / 60
    del fold_data

    best_params = {**tree_params(best, FINAL_ETA), "device": "cuda", "eval_metric": ["rmse", "mae", "mape"]}

    # Pick the number of boosting rounds WITHOUT looking at the test set: hold out the last
    # FINAL_VAL_FRAC of training timesteps (with a gap), early-stop on it, then refit on all
    # training data for that many rounds.
    times_sorted = np.sort(all_times)
    val_start = pd.Timestamp(times_sorted[int(len(times_sorted) * (1 - FINAL_VAL_FRAC))])
    fit_mask = (train["time"] < val_start - pd.Timedelta(days=TUNE_GAP_DAYS)).to_numpy()
    val_mask = (train["time"] >= val_start).to_numpy()
    dfit = xgb.QuantileDMatrix(train.loc[fit_mask, FEATURE_COLS], label=train.loc[fit_mask, TARGET_COL])
    dval = xgb.QuantileDMatrix(train.loc[val_mask, FEATURE_COLS], label=train.loc[val_mask, TARGET_COL], ref=dfit)
    print(f"Choosing number of rounds: early stopping on training timesteps from {val_start.date()} onward")
    es_model = xgb.train(
        best_params,
        dfit,
        num_boost_round=FINAL_MAX_ROUNDS,
        evals=[(dfit, "fit"), (dval, "val")],     # early stopping uses the last eval set (val)
        early_stopping_rounds=FINAL_EARLY_STOP,
        verbose_eval=100,
    )
    n_rounds = es_model.best_iteration + 1
    val_rmse = es_model.best_score
    del dfit, dval, es_model

    with open(RESULTS_PATH, 'a') as outfile:
        outfile.write("### Hyperparameter tuning\n")
        outfile.write(f"- {TUNE_MAX_EVALS} hyperopt trials, {TUNE_NFOLD} time-block CV folds ({TUNE_GAP_DAYS}-day gap), "
                      f"{len(keep)}/{len(all_times)} training timesteps, eta={TUNE_ETA}, {n_gpus} GPU(s), "
                      f"{tune_minutes:.1f} min\n")
        outfile.write(f"- Best CV RMSE: {trials.best_trial['result']['loss']:.4f} Gt/yr\n")
        outfile.write(f"- Best params: max_depth={best_params['max_depth']}, subsample={best_params['subsample']:.3f}, "
                      f"colsample_bytree={best_params['colsample_bytree']:.3f}, "
                      f"min_child_timesteps={best['min_child_timesteps']:.2f}, "
                      f"lambda_timesteps={best['lambda_timesteps']:.3f}, gamma_timesteps={best['gamma_timesteps']:.4f} "
                      f"(x {pixels_per_timestep:,.0f} pixels/timestep); final eta={FINAL_ETA}\n")
        outfile.write(f"- Rounds: {n_rounds} (early stopping on training timesteps from {val_start.date()}, "
                      f"val RMSE {val_rmse:.4f} Gt/yr)\n\n")

    # Refit on all training data for the chosen number of rounds; test is monitored only, never used to stop
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=100,
    )
    return model

def evaluate(model: xgb.Booster, dtest: xgb.QuantileDMatrix, train: pd.DataFrame):
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
    
    y_train = train[TARGET_COL].dropna().to_numpy()
    rmsse = skmetrics.root_mean_squared_scaled_error(truth, preds, y_train)
    mase = skmetrics.mean_absolute_scaled_error(truth, preds, y_train)
    smape = skmetrics.symmetric_mean_absolute_percentage_error(truth, preds)
    
    # TODO: Calculate seasonal (phase, amplitude, etc.) errors

    importance = model.get_score(importance_type="gain")
    importance = pd.Series(importance).sort_values(ascending=False)
    
    # Write metrics to file for easy viewing
    with open(RESULTS_PATH, 'a') as outfile:
        outfile.write(NOTES + "\n\n")
        outfile.write("### Model Metrics\n")
        outfile.write(f"- RMSE   : {rmse:.4f} Gt/yr\n")
        outfile.write(f"- MAE    : {mae:.4f} Gt/yr\n")
        outfile.write(f"- R²     : {r2:.4f}\n")
        outfile.write(f"- RMSSE  : {rmsse:.4f}\n")
        outfile.write(f"- MASE   : {mase:.4f}\n")
        outfile.write(f"- SMAPE  : {smape:.4f}\n")
        outfile.write(f"\n### Top 10 features by gain\n")
        outfile.write(importance.head(10).to_markdown() + "\n\n")

    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsse": rmsse, "mase": mase, "smape": smape, "feature_importance": importance}

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
    ax.set_title(f"Period: {PERIOD}")
    ax.legend()
    xgboost_fig_path = f'/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/figures/{GLACIER_NAME}_xgbresult_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.png'
    # Save figure to .figures/ to upload to GitHub
    plt.savefig(f'./figures/xgboost_result_{PERIOD}.png', dpi=300, bbox_inches='tight')
    plt.savefig(xgboost_fig_path, dpi=300, bbox_inches='tight')
    # Save figure to S3
    fs.put(xgboost_fig_path, f'{S3_BUCKET}/cjense/data/testmodel/figures/{GLACIER_NAME}_xgbresult_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.png')
    plt.close(fig)

def main():
    # Load train and test sets written by preprocess.py
    train = pd.read_parquet(TRAIN_PATH, storage_options=storage_options)
    test = pd.read_parquet(TEST_PATH, storage_options=storage_options)

    # Construct QuantileDMatrices for training and testing (datetime excluded)
    dtrain = xgb.QuantileDMatrix(train[FEATURE_COLS], label=train[TARGET_COL])
    dtest = xgb.QuantileDMatrix(test[FEATURE_COLS], label=test[TARGET_COL], ref=dtrain)

    print("Training new model...")
    model = train_model(train, dtrain, dtest)

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
