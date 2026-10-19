import os
import s3fs

########## CONFIG ##########
GLACIER_NAME  = "zach"
S3_BUCKET     = "s3://gaia"
RESOLUTION    = "MS"                  # MS = monthly on the first of the month
TRAIN_CUTOFF  = "2022-01-01"          # everything before this is train
TEST_START    = "2022-01-01"          # everything from here is test
CACHE_PARQUET = True                  # write flat df to S3 after extraction
MODEL_SEED    = 42                    # random seed to train model
BEESWARM_MAX_POINTS = 200_000         # subsample for the beeswarm plot (plotting all rows takes hours)
SPECIAL_NAME  = "_no_vel"             # include a special signifier to the end of filenames
NOTES         = ("Run with encodings for all feature columns with lags"
                 "and season sin and cosine. Calculating SHAP values of all features using GPU")

FEATURE_COLS = [
    # Spatial vars
    "meltwater", #"ice_velocity",                                   # TODO: Add ice_elevation, distance_to_terminus
    # Non-spatial vars (broadcast)
    "airtemp", "masked_mel_velocity", "melange_area_km2", 
    "ocean_EN4_TFc", "area_km2", "area_change_km2",                 # TODO: Add tongue_length, average_meltwater_runoff, melange_rigidity
    # Lag features
    *[f"{lag}_{var}" for var in ["meltwater", "airtemp", "masked_mel_velocity", "melange_area_km2",
                                 "ocean_EN4_TFc", "area_km2", "area_change_km2"]
      for lag in ("lag_1step", "lag_60d", "lag_90d")],      # no discharge lags (target)
    "roll_30d_mean", "roll_30d_std",
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
    "ocean_EN4_TFc", "area_km2", "area_change_km2",
    # Space
    "x", "y",
]

TARGET_COL = "Discharge (Gt/yr)"

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
TRAIN_PATH = f"{S3_BUCKET}/cjense/data/testmodel/train_{RESOLUTION}{SPECIAL_NAME}.parquet"
TEST_PATH  = f"{S3_BUCKET}/cjense/data/testmodel/test_{RESOLUTION}{SPECIAL_NAME}.parquet"

LOCAL_MODEL_PATH = f"/gpfs/scrubbed/jensencc/negis-seasonality/seasonality-model/models/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.json"
S3_MODEL_PATH    = f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_xgb_{RESOLUTION}_seed{MODEL_SEED}{SPECIAL_NAME}.json"
