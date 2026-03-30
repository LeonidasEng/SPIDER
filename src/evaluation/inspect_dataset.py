import os
import pandas as pd

import sys
print(f"Environment: {sys.executable}")

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
    testset_path = os.path.join(base, "data", "test_sets")
    decision_path = os.path.join(base, "data", "decision")

    FILES = {
            "Observed": "spider_features_obs.parquet",
            "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
            "3 Day Forecast 1230": "spider_features_3day_1230.parquet",
            "3 Day Targets 0030": "spider_targets_3day_0030.parquet",
            "3 Day Targets 1230": "spider_targets_3day_1230.parquet",
            "Testset 0030 LD0": "test_0030_LD0.parquet",
            "Testset 0030 LD1": "test_0030_LD1.parquet",
            "Testset 0030 LD2": "test_0030_LD2.parquet",
            "Testset 1230 LD0": "test_1230_LD0.parquet",
            "Testset 1230 LD1": "test_1230_LD1.parquet",
            "Testset 1230 LD2": "test_1230_LD2.parquet",
            "Debug Rule Layer": "SPIDER_decisions.parquet"
        }
    
    df_obs = pd.read_parquet(os.path.join(dataset_path, FILES["Observed"]))
    df_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 0030"]))
    df_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 1230"]))
    df_tar_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Targets 0030"]))
    df_tar_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Targets 1230"]))


    print(f"Observed Dataset: {df_obs.shape}")
    print(f"3 Day 0030 Feature Dataset: {df_0030.shape}")
    print(f"3 Day 1230 Feature Dataset: {df_1230.shape}")
    print(f"3 Day 0030 Target Dataset: {df_tar_0030.shape}")
    print(f"3 Day 1230 Target Dataset: {df_tar_1230.shape}")

    #df_clim0030 = climateOut(df_tar_0030)
    #df_clim1230 = climateOut(df_tar_1230)
    #df_climGeo  = climateOut(df_tar_geo)

    test_0030_LD0 = pd.read_parquet(os.path.join(testset_path, FILES["Testset 0030 LD0"]))
    test_0030_LD1 = pd.read_parquet(os.path.join(testset_path, FILES["Testset 0030 LD1"]))
    test_0030_LD2 = pd.read_parquet(os.path.join(testset_path, FILES["Testset 0030 LD2"]))

    test_1230_LD0 = pd.read_parquet(os.path.join(testset_path, FILES["Testset 1230 LD0"]))
    test_1230_LD1 = pd.read_parquet(os.path.join(testset_path, FILES["Testset 1230 LD1"]))
    test_1230_LD2 = pd.read_parquet(os.path.join(testset_path, FILES["Testset 1230 LD2"]))

    debug_decision = pd.read_parquet(os.path.join(decision_path, FILES["Debug Rule Layer"]))

    # RECOMMENDED: If in VSCode, recommend using Data Wrangler extension to view the datasets.

    print("stop") # Put BREAKPOINT HERE FOR DEBUG OR NOTHING WILL HAPPEN
    
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

    # POST ALL DATA (2012-2025)
    # Observed Dataset: (38202, 11)
    # 3 Day 0030 Feature Dataset: (113465, 15)
    # 3 Day 1230 Feature Dataset: (94455, 15)
    # 3 Day 0030 Target Dataset: (113465, 19)
    # 3 Day 1230 Target Dataset: (94455, 19)

    # Dataset    | Old rows | New rows  | Increase
    # 3-Day 0030 | 23,117   | 113,465   | +90,348 (~391%)
    # 3-Day 1230 | 19,026   |  94,455   | +75,429 (~396%)

if __name__ == "__main__":
    main()