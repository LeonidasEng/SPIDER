import os
import pandas as pd

TARGETS = {
        "3 Day Forecast 0030": "spider_targets_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_targets_3day_1230.parquet",
        "Geomag Forecast": "spider_targets_geomag.parquet"
    }

# Create Test and Train datasets: https://www.geeksforgeeks.org/python/pandas-create-test-and-train-samples-from-dataframe/

def dataSplit(df:pd.DataFrame):
    # Split into 70-30 split
    df = df.sort_index()

    split_idx = int(len(df) * 0.7)

    train_set = df[:split_idx]
    test_set = df[split_idx:]

    print(f"Train: {train_set.shape}, Test: {test_set.shape}")
    return train_set, test_set

def persistenceBase(df:pd.DataFrame, target_col="is_large_error"):
    df = df.copy()

    df["pred_persistence"] = df[target_col].shift(1)

    df = df.dropna(subset=["pred_persistence"])

    y_true = df[target_col]
    y_pred = df["pred_persistence"]

    accuracy = (y_true == y_pred).mean()

    print(f"Persistence accuracy {accuracy: .3f}")

    return y_true, y_pred

def main():
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    dataset_path = os.path.join(base, "data", "datasets")
    
    train_sets = {}
    test_sets = {}

    for dataset, file_name in TARGETS.items():

        df = pd.read_parquet(os.path.join(dataset_path, file_name))
        train_set, test_set = dataSplit(df)

        train_sets[dataset] = train_set
        test_sets[dataset]  = test_set

    train3D_morn = train_sets["3 Day Forecast 0030"]
    train3D_aft = train_sets["3 Day Forecast 1230"] 
    train_geo = train_sets["Geomag Forecast"]

    test3D_morn = test_sets["3 Day Forecast 0030"]
    test3D_aft = test_sets["3 Day Forecast 1230"]
    test_Geo = test_sets["Geomag Forecast"]
    
    y_true, y_pred = persistenceBase(test3D_morn)
    print("stop")

if __name__ == "__main__":
    main()