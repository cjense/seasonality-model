import os
import s3fs

########## CONFIG ##########
GLACIER_NAME  = "zach"
S3_BUCKET     = "s3://gaia"
RESOLUTION    = "6D"                  # MS = monthly on the first of the month
CACHE_PARQUET = True                 # write flat df to S3 after extraction
MODEL_SEED    = 42                    # random seed to train model
BEESWARM_MAX_POINTS = 200_000         # subsample for the beeswarm plot (plotting all rows takes hours)
SPECIAL_NAME  = "_no_vel"             # include a special signifier to the end of filenames
NOTES         = ("Run with encodings for all feature columns with lags "
                 "and season sin and cosine. Test 2000-2015 and 2015-2021 with max_depth 5, no area_change_km2. "
                 "Running on pre-2015 and post-2015 data. "
                 "Calculating SHAP values of all features using GPU.")

FEATURE_COLS = [
    # Spatial vars
    "meltwater", #"ice_velocity",                                   # TODO: Add ice_elevation, distance_to_terminus
    # Non-spatial vars (broadcast)
    "airtemp", "masked_mel_velocity", "melange_area_km2", 
    "ocean_EN4_TFc", "area_km2",               # TODO: Add tongue_length, average_meltwater_runoff, melange_rigidity
    # Lag features
    *[f"{lag}_{var}" for var in ["meltwater", "airtemp", "masked_mel_velocity", "melange_area_km2",
                                 "ocean_EN4_TFc", "area_km2"]
      for lag in ("lag_1step", "lag_60d", "lag_90d")],      # no discharge lags (target)
    # "roll_30d_mean", "roll_30d_std",
    # Time encodings
    "season_sin", "season_cos",
    # "year_norm", "time_days",
    # Space
    "x", "y",
]

NON_SEASONAL_VARS = [
    # Spatial vars
    "meltwater", #"ice_velocity",
    # Non-spatial vars
    "airtemp", "masked_mel_velocity", "melange_area_km2",
    "ocean_EN4_TFc", "area_km2",
    # Space
    "x", "y",
]

TARGET_COL = "Discharge (Gt/yr)"

########## Time periods ##########
# Each period is trained and evaluated separately: train on [start, train_cutoff),
# test on [train_cutoff, end]. Pick one with the PERIOD environment variable,
# e.g. `PERIOD=pre2015 python run_xgboost.py`. Defaults to the full record.
PERIODS = {
    "full":     {"start": "2000-01-01", "train_cutoff": "2018-01-01", "end": "2020-12-31"},
    "pre2015":  {"start": "2000-01-01", "train_cutoff": "2012-01-01", "end": "2014-12-31"},
    "post2015": {"start": "2015-01-01", "train_cutoff": "2018-06-01", "end": "2020-12-31"},
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

LOCAL_MODEL_PATH = f"/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/models/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.json"
S3_MODEL_PATH    = f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{RUN_NAME}.json"

os.makedirs("./figures", exist_ok=True)
