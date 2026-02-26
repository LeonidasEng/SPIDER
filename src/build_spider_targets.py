import os 
import pandas as pd

FILES = {
        "Observed": "spider_features_obs.parquet",
        "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_features_3day_1230.parquet",
        #"Geomag Forecast": "spider_features_geomag.parquet"
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
    df["is_severe_error"] = (df["abs_delta_kp"] > 2)

    return df

def buildTargetsT2(dataset_path:str):
    pass

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
        
        df_pack[dataset] = buildTargetsT1(dataset_path, file_name)

    # ds_3day_0030, ds_3day_1230, ds_geomag = (df_pack["3 Day Forecast 0030"], df_pack["3 Day Forecast 1230"], 
    #                                         df_pack["Geomag Forecast"])
    ds_3day_0030, ds_3day_1230 = (df_pack["3 Day Forecast 0030"], df_pack["3 Day Forecast 1230"])
   
    data_output_path = dataset_path
    os.makedirs(data_output_path, exist_ok=True)

    path_3day_morn = os.path.join(data_output_path, "spider_targets_3day_0030.parquet")
    path_3day_aft = os.path.join(data_output_path, "spider_targets_3day_1230.parquet")
    #path_geomag = os.path.join(data_output_path, "spider_targets_geomag.parquet")
    
    ds_3day_0030.to_parquet(path_3day_morn)
    print(f"SPIDER 3 Day 0030 target parquet was saved to: {path_3day_morn}") 
    ds_3day_1230.to_parquet(path_3day_aft)
    print(f"SPIDER 3 Day 1230 feature parquet was saved to: {path_3day_aft}")
    #ds_geomag.to_parquet(path_geomag)
    #print(f"SPIDER Geomag feature parquet was saved to: {path_geomag}")

if __name__ == "__main__":
    main()