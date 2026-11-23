import os
import sys
import json
import s3fs

########## CONFIG ##########
GLACIER_NAME  = "zach"
S3_BUCKET     = "s3://gaia"
RESOLUTION    = "6D"                  # MS = monthly on the first of the month
CACHE_PARQUET = True                 # write flat df to S3 after extraction
MODEL_SEED    = 42                    # random seed to train model
BEESWARM_MAX_POINTS = 200_000         # subsample for the beeswarm plot (plotting all rows takes hours)
SHAP_CHUNK_ROWS     = 1_000_000       # rows per GPU TreeSHAP call (large single calls overflow the CUDA launch config)
GEOSHAP_MAX_ROWS    = 500_000        # GeoShapley row budget: whole timesteps (all pixels) are sampled up to this many rows
SPECIAL_NAME  = "_2008_2025"             # include a special signifier to the end of filenames
NOTES         = ("Run on ZI 2008-2025 using cross-validation and hyperpaarameter tuning. Velocity as target. 6D resolution.")

FEATURE_COLS = [
    # Spatial vars
    "Per-pixel Meltwater Runoff (mmWE/day)", "Basin-Averaged Velocity (m/yr)",               # TODO: Add ice_elevation, distance_to_terminus
    # Non-spatial vars
    "Air Temperature (C)", "Mélange Area (km^2)",                           # TODO: Add tongue_length, average_meltwater_runoff, melange_rigidity
    "Ocean Thermal Forcing (C)", "Terminus Area (km^2)",
    "Basin-Averaged Meltwater Runoff (mmWE/day)",
    # Lag features
    # *[f"{lag}_{var}" for var in ["Per-pixel Meltwater Runoff (mmWE/day)",
    #                              "Air Temperature (C)", "Mélange Area (km^2)",
    #                              "Ocean Thermal Forcing (C)", "Terminus Area (km^2)",
    #                              "Basin-Averaged Meltwater Runoff (mmWE/day)"]
    #   for lag in ("lag_1step", "lag_60d", "lag_90d")],      # no discharge lags (target)
    # "roll_30d_mean", "roll_30d_std",
    # Time encodings
    # "season_sin", "season_cos",
    # "year_norm", "time_days",
    # Space
    # "x", "y",
]

TARGET_COL = "Near-Terminus Velocity (m/yr)"#"Discharge (Gt/yr)"

########## Feature selection / hyperparameter tuning ##########
# When True, the model uses the features and XGBoost hyperparameters chosen by select_features.py
# (blocked cross-validation, below) instead of FEATURE_COLS and DEFAULT_PARAMS.
# Run select_features.py first for each PERIOD; it saves its choice to SELECTION_PATH (below).
USE_FEATURE_SELECTION = True

########## Cross-validation ##########
# Blocked CV over the whole period (no fixed train/test split): the record is cut into chunks
# of CV_BLOCK_YEARS calendar years. Each fold holds out one chunk and trains on all the others,
# before and after it. Used for feature selection/tuning and for the reported RMSE.
CV_BLOCK_YEARS = 1                     # chunk length; 1 = hold out one full seasonal cycle per fold
CV_GAP_MONTHS  = 3                     # months on each side of the held-out chunk dropped from training,
                                       # so autocorrelation/lag features don't leak across the boundary

# Hand-picked hyperparameters, used when USE_FEATURE_SELECTION is False and as the tuning baseline.
# sklearn/XGBRegressor names. n_estimators is fixed (no early stopping, which would need a validation set).
DEFAULT_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
)

########## Detrending ##########
# Remove the long-term trend from the target and every input feature before training, so the
# model (and SHAP) attribute the seasonal/interannual signal rather than the multi-year trend.
# A polynomial in time is fit over the whole period and subtracted; the mean is added back,
# so values stay in physical units.
DETREND        = True
DETREND_DEGREE = 1                     # 1 = linear trend; 2 = quadratic (extrapolates less safely)
# Columns that are encodings of time/space, not physical variables: never detrended
DETREND_EXCLUDE = ["season_sin", "season_cos", "year_norm", "time_days", "x", "y"]

# Columns that vary by pixel. If no feature uses one of these, the model is non-spatial:
# every pixel at a timestep has identical features and target, so preprocess.py builds
# one row per timestep from the non-spatial parquet instead of the (huge) per-pixel table.
SPATIAL_COLS = ["Per-pixel Meltwater Runoff (mmWE/day)", "ice_velocity", "x", "y"]

########## Compute / output locations ##########
# XGBoost device: "cuda" on the Linux GPU cluster, "cpu" elsewhere (override with XGB_DEVICE)
DEVICE = os.environ.get("XGB_DEVICE", "cuda" if sys.platform.startswith("linux") else "cpu")

# Where models and full-resolution figures are written (override with OUTPUT_DIR).
# Defaults to the Tillicum scrubbed directory when it exists, otherwise this repo.
_CLUSTER_OUTPUT_DIR = "/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                            _CLUSTER_OUTPUT_DIR if os.path.isdir(_CLUSTER_OUTPUT_DIR) else os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")

########## Time periods ##########
# Each period is modeled and cross-validated separately over [start, end]. Pick one with the
# PERIOD environment variable, e.g. `PERIOD=pre2015 python run_xgboost.py`. Defaults to the full record.

# Air T started 2008-07-17
PERIODS = {
    "full":     {"start": "2008-07-17", "end": "2025-12-31"},
    # "pre2015":  {"start": "2008-07-17", "end": "2014-12-31"},
    # "post2015": {"start": "2015-01-01", "end": "2025-12-31"},
}
PERIOD = os.environ.get("PERIOD", "full")
if PERIOD not in PERIODS:
    raise ValueError(f"Unknown PERIOD '{PERIOD}'. Choose from {list(PERIODS)}.")
PERIOD_START = PERIODS[PERIOD]["start"]
PERIOD_END   = PERIODS[PERIOD]["end"]

# Appended to every per-period output (dataset, models, figures, SHAP values).
# The flat feature cache and correlation matrix are shared, so they keep SPECIAL_NAME only.
RUN_NAME = f"{SPECIAL_NAME}{'_detrended' if DETREND else ''}_{PERIOD}"

# Results for each period are written to their own markdown file
RESULTS_PATH = f"modelresults_{PERIOD}.md"

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

########## Paths shared between scripts ##########
# All rows of the period (time, features, target); CV folds are made from it on the fly
DATA_PATH = f"{S3_BUCKET}/cjense/data/testmodel/data_{RESOLUTION}{RUN_NAME}.parquet"

LOCAL_MODEL_PATH = f"{MODEL_DIR}/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.json"
S3_MODEL_PATH    = f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.json"

# Features + hyperparameters chosen by select_features.py for this run
SELECTION_PATH = f"{S3_BUCKET}/cjense/data/testmodel/selection_{GLACIER_NAME}_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.json"

########## Resolve which features/params this run uses ##########
MANUAL_FEATURE_COLS = list(FEATURE_COLS)      # the hand-picked list above, kept for comparison
MODEL_PARAMS = dict(DEFAULT_PARAMS)
SELECTION = None                              # the loaded select_features.py output, if any
if USE_FEATURE_SELECTION:
    try:
        with fs.open(SELECTION_PATH, "r") as f:
            SELECTION = json.load(f)
        FEATURE_COLS = SELECTION["features"]
        MODEL_PARAMS = SELECTION["params"]
    except FileNotFoundError:
        pass                                  # require_selection() raises in scripts that need it

def require_selection():
    """Fail early in pipeline scripts when USE_FEATURE_SELECTION is on but select_features.py hasn't run."""
    if USE_FEATURE_SELECTION and SELECTION is None:
        raise FileNotFoundError(
            f"USE_FEATURE_SELECTION is True but no selection was found at {SELECTION_PATH}. "
            f"Run `PERIOD={PERIOD} python select_features.py` first, or set USE_FEATURE_SELECTION = False.")

# Columns that vary by pixel. If no feature uses one of these, the model is non-spatial:
# every pixel at a timestep has identical features and target, so preprocess.py builds
# one row per timestep from the non-spatial parquet instead of the (huge) per-pixel table.
# Match whole names (or lag/rolling features built from them, e.g. "lag_1step_x"), not substrings:
# a substring test flags "Basin-Averaged Meltwater Runoff (mmWE/day)" as spatial because it contains "y".
NONSPATIAL_ONLY = not any(f == s or f.endswith("_" + s) for f in FEATURE_COLS for s in SPATIAL_COLS)

os.makedirs("./figures", exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
