import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

TARGETS = {
        "3 Day Forecast 0030": "spider_targets_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_targets_3day_1230.parquet",
        "Geomag Forecast": "spider_targets_geomag.parquet"
    }

# Create Test and Train datasets: https://www.geeksforgeeks.org/python/pandas-create-test-and-train-samples-from-dataframe/

def dataSplit(df:pd.DataFrame):
    # Split into 70-30 split
    df = df.sort_index()

    split_idx = int(len(df) * 0.7) # 70% of the data

    train_set = df[:split_idx] # Up to 70%
    test_set = df[split_idx:] # Remaining 30%

    print(f"Train: {train_set.shape}, Test: {test_set.shape}")
    return train_set, test_set

def persistenceBase(df:pd.DataFrame, lead_day:int, target_col="is_large_error"):
    df = df.copy()

    df = df[df["lead_day"] == lead_day]

    df["pred_persistence"] = df[target_col].shift(1) # Is tomorrow same as today?

    df = df.dropna(subset=["pred_persistence"])

    y_true = df[target_col].astype(int)
    y_pred = df["pred_persistence"].astype(int)

    accuracy = (y_true == y_pred).mean()

    print(f"Persistence accuracy {accuracy: .3f}")

    return y_true, y_pred

def confusionMatrix(observed, predicted):
    
    confusion_matrix = metrics.confusion_matrix(observed, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.show()

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
        
        for lead_day in [0, 1, 2]:
            y_true, y_pred = persistenceBase(test_set, lead_day)
            print(y_true.dtype, y_pred.dtype)
            confusionMatrix(y_true, y_pred)
            

    print("stop")

if __name__ == "__main__":
    main()