import os
import pandas as pd

# Debugger for all parquet files to quickly assess datasets.

def main():
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    dataset_path = os.path.join(base, "data", "datasets")

    FILES = {
            "Observed": "spider_features_obs.parquet",
            "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
            "3 Day Forecast 1230": "spider_features_3day_1230.parquet",
            "Geomag Forecast": "spider_features_geomag.parquet",
            "3 Day Targets 0030": "spider_targets_3day_0030.parquet",
            "3 Day Targets 1230": "spider_targets_3day_1230.parquet",
            "Geomag Targets": "spider_targets_geomag.parquet"
        }
    
    df_obs = pd.read_parquet(os.path.join(dataset_path, FILES["Observed"]))
    df_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 0030"]))
    df_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 1230"]))
    df_geo = pd.read_parquet(os.path.join(dataset_path, FILES["Geomag Forecast"]))
    df_tar_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Targets 0030"]))
    df_tar_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Targets 1230"]))
    df_tar_geo = pd.read_parquet(os.path.join(dataset_path, FILES["Geomag Targets"]))

    print("stop") # For debug
    

if __name__ == "__main__":
    main()