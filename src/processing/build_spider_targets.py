import os 
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

FILES = {
        "Observed": "spider_features_obs.parquet",
        "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_features_3day_1230.parquet"
    }

def buildTargetsT1(dataset_path:str, file_name:str):
    '''
    Tier 1 Targets for Modelling
    '''
    df = pd.read_parquet(os.path.join(dataset_path, file_name)).copy()

    forecast_col = [col for col in df.columns if col.startswith("kp_") and col != "kp_obs"]
    
    kp_forecast = forecast_col[0]

    df["delta_kp"] = df[kp_forecast] - df["kp_obs"]
    df["abs_delta_kp"] = df["delta_kp"].abs()

    df["is_large_error"] = (df["abs_delta_kp"] > 1)

    return df

def buildTargetsT2(dataset_path:str, file_name:str):
    '''
    Tier 2 Targets for Modelling
    '''
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
    pass

def main():
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