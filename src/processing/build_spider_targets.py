import os 
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# This script requires updated feature parquet files in order to generate targets.

FILES = {
        "Observed": "spider_features_obs.parquet",
        "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_features_3day_1230.parquet"
    }

def buildTargetsT1(dataset_path:str, file_name:str) -> pd.DataFrame:
    """
    Builds Tier 1 modelling targets from a SPIDER feature dataset.

    Args:
        dataset_path (str): Directory containing SPIDER parquet files.
        file_name (str): Name of the feature parquet file to process.
    
    Returns:
        pandas.DataFrame:
            Feature dataset with added forecast error target columns.
    
    Notes:
        Tier 1 targets include:
        - `delta_kp`
        - `abs_delta_kp`
        - `is_large_error`: A large forecast error is defined as an 
        absolute Kp error greater than 1.
    """
    df = pd.read_parquet(os.path.join(dataset_path, file_name)).copy()

    forecast_col = [col for col in df.columns if col.startswith("kp_") and col != "kp_obs"]
    
    kp_forecast = forecast_col[0]

    df["delta_kp"] = df[kp_forecast] - df["kp_obs"]
    df["abs_delta_kp"] = df["delta_kp"].abs()

    df["is_large_error"] = (df["abs_delta_kp"] > 1)

    return df

def buildTargetsT2(dataset_path:str, file_name:str):
    """
    Builds Tier 2 modelling targets from a SPIDER feature dataset.

    Args:
        dataset_path (str): Directory containing SPIDER feature parquet files.
        file_name (str): Name of the feature parquet file to process.

    Returns:
        pandas.DataFrame:
            Feature dataset with added large error targets.

    Notes:
        Tier 2 targets include both instantaneous and windowed large-error targets.

        An instantaneous large error is defined as:
        `abs(forecast_kp - observed_kp) > 1`

        The windowed large-error target also marks neighbouring 3-hour bins
        as `True` when a large error occurs. This reduces double penalties caused
        by small timing offsets around clustered forecast errors.

        Intermediate calculation columns are removed before returning the final target dataset.
    """
    df = pd.read_parquet(os.path.join(dataset_path, file_name)).copy()

    forecast_col = [col for col in df.columns if col.startswith("kp_") and col != "kp_obs"]
    
    kp_forecast = forecast_col[0]

    df["delta_kp"] = df[kp_forecast] - df["kp_obs"]
    df["abs_delta_kp"] = df["delta_kp"].abs() # Absolute value of difference

    df["is_large_error"] = (df["abs_delta_kp"] > 1) # Target

    # Ensure correct order
    df:pd.DataFrame = df.sort_values(["lead_day", "issue_time_utc", "valid_start_utc"])

    # Adding windowed target 3 hour tolerance based on (Owens:2018):
    prev_err = df["is_large_error"].shift(1).fillna(0)
    current_err = df["is_large_error"]
    next_err = df["is_large_error"].shift(-1).fillna(0)

    # This will prevent double penalties, bool type to preve
    df["is_large_error_win"] = ((prev_err == 1) | (current_err == 1) | (next_err == 1)).astype(bool)
    
    # Return previous order
    df = df.sort_values(["issue_time_utc", "valid_start_utc"])
    
    # Diagnostics
    # print("Correlation with Kp Observed values:")
    # corr_obs_ld = df.groupby("lead_day").apply(
    #     lambda x: x[["prev_error", "error_rate_24h", "time_since_last_error"]].corrwith(x["kp_obs"])
    # )
    # print(corr_obs_ld)

    # print("Correlation with is_large_error_win:")
    # corr_err_ld = df.groupby("lead_day").apply(
    #     lambda x: x[["prev_error", "error_rate_24h", "time_since_last_error"]].corrwith(x["is_large_error_win"])
    # )
    # print(corr_err_ld)

    # These columns are only needed for calculation and can safely be removed
    df = df.drop(columns=["kp_obs", "delta_kp", "abs_delta_kp"])

    return df

def buildTargetsT3(dataset_path:str):
    """
    Placeholder for future Tier 3 target generation.

    Args:
        dataset_path (str): Directory containing SPIDER feature parquet files.
    """
    pass

def main():
    """
    Main entry point for building SPIDER modelling target datasets.

    The pipeline:
    - Locates SPIDER feature parquet files.
    - Skips the observation-only feature dataset.
    - Builds target labels for the 0030 and 1230 3-day forecast datasets.
    - Applies the Tier 2 windowed large error target strategy (currently).
    - Writes final target datasets to parquet files.

    Notes:
        This script requires the SPIDER feature parquet files to exist before target generation can run.

        Output target parquet files are written to:
        `data/datasets/`
    
    Raises:
        EnvironmentError: If the `SPIDER` environment variable is not defined.
    """
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    df_pack = {}

    dataset_path = os.path.join(base, "data", "datasets")

    for dataset, file_name in FILES.items():
        
        if dataset == "Observed":
            continue
        
        df_pack[dataset] = buildTargetsT2(dataset_path, file_name) # Previously T1

    ds_3day_0030, ds_3day_1230 = (df_pack["3 Day Forecast 0030"], df_pack["3 Day Forecast 1230"])
   
    data_output_path = dataset_path
    os.makedirs(data_output_path, exist_ok=True)

    path_3day_morn = os.path.join(data_output_path, "spider_targets_3day_0030.parquet")
    path_3day_aft = os.path.join(data_output_path, "spider_targets_3day_1230.parquet")
    
    ds_3day_0030.to_parquet(path_3day_morn)
    print(f"SPIDER 3 Day 0030 target parquet was saved to: {path_3day_morn}") 
    ds_3day_1230.to_parquet(path_3day_aft)
    print(f"SPIDER 3 Day 1230 feature parquet was saved to: {path_3day_aft}")

if __name__ == "__main__":
    main()