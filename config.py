import os
import sys
import s3fs

########## CONFIG ##########
GLACIER_NAME  = "zach"
S3_BUCKET     = "s3://gaia"
RESOLUTION    = "MS"                  # MS = monthly on the first of the month
CACHE_PARQUET = True                 # write flat df to S3 after extraction
MODEL_SEED    = 42                    # random seed to train model
BEESWARM_MAX_POINTS = 200_000         # subsample for the beeswarm plot (plotting all rows takes hours)
GEOSHAP_MAX_ROWS    = 500_000         # GeoShapley row budget: whole timesteps (all pixels) are sampled up to this many rows
SPECIAL_NAME  = "_2008_2015_melt_airT_OTF"             # include a special signifier to the end of filenames
NOTES         = ("Run on ZI 2008-2025 with meltwater, air T, ocean thermal forcing. Monthly resolution.")

FEATURE_COLS = [
    # Spatial vars
    # "Per-pixel Meltwater Runoff (mmWE/day)", #"ice_velocity",               # TODO: Add ice_elevation, distance_to_terminus
    # Non-spatial vars
    "Air Temperature (C)", #"Mélange Area (km^2)",                           # TODO: Add tongue_length, average_meltwater_runoff, melange_rigidity
    "Ocean Thermal Forcing (C)", #"Terminus Area (km^2)",
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

TARGET_COL = "Discharge (Gt/yr)"

# Columns that vary by pixel. If no feature uses one of these, the model is non-spatial:
# every pixel at a timestep has identical features and target, so preprocess.py builds
# one row per timestep from the non-spatial parquet instead of the (huge) per-pixel table.
SPATIAL_COLS = ["Per-pixel Meltwater Runoff (mmWE/day)", "ice_velocity", "x", "y"]
NONSPATIAL_ONLY = not any(s in f for f in FEATURE_COLS for s in SPATIAL_COLS)

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
# Each period is trained and evaluated separately: train on [start, train_cutoff),
# test on [train_cutoff, end]. Pick one with the PERIOD environment variable,
# e.g. `PERIOD=pre2015 python run_xgboost.py`. Defaults to the full record.
PERIODS = {
    "full":     {"start": "2008-07-17", "train_cutoff": "2018-01-01", "end": "2025-12-31"},
    # "pre2015":  {"start": "2008-07-17", "train_cutoff": "2013-04-01", "end": "2014-12-31"},
    # "post2015": {"start": "2015-01-01", "train_cutoff": "2022-06-01", "end": "2025-12-31"},
}
PERIOD = os.environ.get("PERIOD", "full")
if PERIOD not in PERIODS:
    raise ValueError(f"Unknown PERIOD '{PERIOD}'. Choose from {list(PERIODS)}.")
PERIOD_START = PERIODS[PERIOD]["start"]
TRAIN_CUTOFF = PERIODS[PERIOD]["train_cutoff"]     # everything before this is train
TEST_START   = TRAIN_CUTOFF                        # everything from here is test
PERIOD_END   = PERIODS[PERIOD]["end"]

# Appended to every per-period output (train/test sets, models, figures, SHAP values).
# The flat feature cache and correlation matrix are shared, so they keep SPECIAL_NAME only.
RUN_NAME = f"{SPECIAL_NAME}_{PERIOD}"

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
TRAIN_PATH = f"{S3_BUCKET}/cjense/data/testmodel/train_{RESOLUTION}{RUN_NAME}.parquet"
TEST_PATH  = f"{S3_BUCKET}/cjense/data/testmodel/test_{RESOLUTION}{RUN_NAME}.parquet"

LOCAL_MODEL_PATH = f"{MODEL_DIR}/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.json"
S3_MODEL_PATH    = f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.json"

os.makedirs("./figures", exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
