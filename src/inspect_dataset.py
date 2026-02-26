import os
import pandas as pd

# Debugger for all parquet files to quickly assess datasets.

def climateOut(df): 
    cols = ["bz_gsm", "v_sw", "ey", "np", "pdyn", "beta", "mach_alfven"]
    return df[cols].copy()

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
            #"Geomag Forecast": "spider_features_geomag.parquet",
            "3 Day Targets 0030": "spider_targets_3day_0030.parquet",
            "3 Day Targets 1230": "spider_targets_3day_1230.parquet",
            #"Geomag Targets": "spider_targets_geomag.parquet"
        }
    
    df_obs = pd.read_parquet(os.path.join(dataset_path, FILES["Observed"]))
    df_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 0030"]))
    df_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 1230"]))
    #df_geo = pd.read_parquet(os.path.join(dataset_path, FILES["Geomag Forecast"]))
    df_tar_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Targets 0030"]))
    df_tar_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Targets 1230"]))
    #df_tar_geo = pd.read_parquet(os.path.join(dataset_path, FILES["Geomag Targets"]))

    print(f"Observed Dataset: {df_obs.shape}")
    print(f"3 Day 0030 Feature Dataset: {df_0030.shape}")
    print(f"3 Day 1230 Feature Dataset: {df_1230.shape}")
    #print(f"Geomag Feature Dataset: {df_geo.shape}")
    print(f"3 Day 0030 Target Dataset: {df_tar_0030.shape}")
    print(f"3 Day 1230 Target Dataset: {df_tar_1230.shape}")
    #print(f"Geomag Target Dataset: {df_tar_geo.shape}")

    #df_clim0030 = climateOut(df_tar_0030)
    #df_clim1230 = climateOut(df_tar_1230)
    #df_climGeo  = climateOut(df_tar_geo)

    print("stop") # For debug
    
    # ORIGINAL DATA
    # Observed Dataset: (11675, 11)
    # 3 Day 0030 Feature Dataset: (23117, 15)
    # 3 Day 1230 Feature Dataset: (19026, 15)
    # Geomag Feature Dataset: (20726, 15)
    # 3 Day 0030 Target Dataset: (23117, 19)
    # 3 Day 1230 Target Dataset: (19026, 19)
    # Geomag Target Dataset: (20726, 19)

    # POST NEW DATA (2022-2025) 
    # Observed Dataset: (11675, 11)
    # 3 Day 0030 Feature Dataset: (30578, 15)
    # 3 Day 1230 Feature Dataset: (25215, 15)
    # Geomag Feature Dataset: (20726, 15)
    # 3 Day 0030 Target Dataset: (30578, 19)
    # 3 Day 1230 Target Dataset: (25215, 19)
    # Geomag Target Dataset: (20726, 19)

    # Dataset    | Old rows | New rows | Increase
    # 3-Day 0030 | 23,117   | 30,578   | +7,461 (~32%)
    # 3-Day 1230 | 19,026   | 25,215   | +6,189 (~32%)
    # Geomag     | 20,726   | 20,726   | no change

    # POST LARGE DATA (2015-2025)
    # Observed Dataset: (31898, 11)
    # 3 Day 0030 Feature Dataset: (94873, 15)
    # 3 Day 1230 Feature Dataset: (78990, 15)
    # 3 Day 0030 Target Dataset: (94873, 19)
    # 3 Day 1230 Target Dataset: (78990, 19)

if __name__ == "__main__":
    main()