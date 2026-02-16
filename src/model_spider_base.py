import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

TARGETS = {
        "3 Day Forecast 0030": "spider_targets_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_targets_3day_1230.parquet",
        "Geomag Forecast": "spider_targets_geomag.parquet"
    }

# Create Test and Train datasets: https://www.geeksforgeeks.org/python/pandas-create-test-and-train-samples-from-dataframe/
# I'll add later when I need for NB and LR

def dataSplit(df:pd.DataFrame, dataset:str):
    # Split into 70-30 split
    df = df.sort_index()

    split_idx = int(len(df) * 0.7) # 70% of the data

    train_set = df[:split_idx] # Up to 70%
    test_set = df[split_idx:] # Remaining 30%

    print(f"{dataset}")
    print(f"Train: {train_set.shape}, Test: {test_set.shape}")
    return train_set, test_set

def persistenceBase(df:pd.DataFrame, lead_day:int, target_col="is_large_error"):
    '''
    Was forecast wrong today? Will it be wrong tomorrow?
    '''
    df = df.copy()

    df = df[df["lead_day"] == lead_day]

    df["probability"] = df[target_col].shift(1)

     # Remove missing values
    df = df.dropna(subset=["probability"])

    y_true = df[target_col].astype(int)         # Ground Truth
    y_prob = df["probability"].astype(float)    # Deterministic 0 or 1 but treating as prob for Brier and ROC
    y_pred = (y_prob >= 1).astype(int)        # Predicted binary classification

    # accuracy = (y_true == y_pred).mean()
    # print(f"Persistence accuracy {accuracy: .3f}")

    return y_true, y_prob, y_pred

def climatologyBase(df:pd.DataFrame, lead_day:int, target_col="is_large_error", forecast_col="kp_forecast"):
    df = df.copy()

    df = df[df["lead_day"] == lead_day]

    # # Using Data Wrangler, I can see the distribution of OMNI data.
    # # Using only 2 drivers to maintain broad outlook (I think!)
    # bins_bz = [-np.inf, -20, -10, -5, 0, 5, np.inf] 
    # # [-inf, -20], [-20, -10], [-10, -5], [-5, 0], [0, 5], [5, inf]
    # bins_vsw = [0, 350, 450, 600, 800, np.inf] 
    # # [0, 350], [350, 450], [450, 600], [600, 800], [800, inf]

    # df["bz_bin"]  = pd.cut(df["bz_gsm"], bins=bins_bz, include_lowest=True) # Include lowest for Large -Bz = Storm
    # df["vsw_bin"] = pd.cut(df["v_sw"], bins=bins_vsw)

    # Climatology-based probabilty model P(|ΔKp| > 1)| Forecast Kp)
    climatology = (
        df.groupby(forecast_col)[target_col]
        .mean()
        .rename("probability")
        .reset_index()
    )

    df = df.merge(climatology, on=forecast_col, how="left") # ["bz_bin", "vsw_bin"]
    
    y_true = df[target_col].astype(int)
    y_prob = df["probability"].fillna(df[target_col].mean()) # Fill missing with zeros
    y_pred = (y_prob >= 0.5).astype(int) # baseline prediction threshold (Camporeale:2025)

    # accuracy = (y_true == y_pred).mean()
    # print(f"Climatology accuracy: {accuracy: .3f}")

    return y_true, y_prob, y_pred

def cmDisplay(observed, predicted):
    
    confusion_matrix = metrics.confusion_matrix(observed, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.show()

def metricsTable(dataset, lead_day, model_name, y_true, y_prob, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    
    brier = metrics.brier_score_loss(y_true, y_prob) # Calibration 
    auc = metrics.roc_auc_score(y_true, y_prob) # Discrimination
    
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    pod = TP / (TP+FN) # Probability of Detection
    # Fraction of real large forecast errors that were successfully detected
    far = FP / (TP+FP) # False Alarm Ratio
    # Fraction of unnecessary warnings 
    csi = TP / (TP+FP+FN) # Critical Success Index
    # Success rate ignoring correct non-events
    tss = (TP/(TP+FN)) - (FP/(FP+TN)) # True Skill Score 
    # Ability to separate bad forecasts from good forecasts
    hss = 2*(TP*TN - FN*FP) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)) # Heidke Skill Score
    # Forecast skill relative to random chance

    # Return a row of metric results (round to 2dp)
    return {
        "Dataset": dataset,
        "Model": model_name,
        "Lead Day": lead_day,
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
        "Brier": round(brier, 2) ,
        "AUC": round(auc, 2),
        "POD": round(pod, 2),
        "FAR": round(far, 2),
        "CSI": round(csi, 2),
        "TSS": round(tss, 2),
        "HSS": round(hss, 2)
    }

def main():
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    dataset_path = os.path.join(base, "data", "datasets")
    
    train_sets = {}
    test_sets = {}
    tables = {}

    kp_column = {
            "3 Day Forecast 0030": "kp_threeday",
            "3 Day Forecast 1230": "kp_threeday",
            "Geomag Forecast": "kp_geomag"
        }
    
    for dataset, file_name in TARGETS.items():

        df = pd.read_parquet(os.path.join(dataset_path, file_name))
        train_set, test_set = dataSplit(df, dataset)

        train_sets[dataset] = train_set
        test_sets[dataset]  = test_set
        tables[dataset] = {}
        
        for lead_day in [0, 1, 2]:
            rows = []
            print(f"Dataset: {dataset}")
            y_true, y_prob,  y_pred = persistenceBase(test_set, lead_day)
            print(f"{y_true.shape}, {y_pred.shape}")
            rows.append(metricsTable(dataset, lead_day, "Persistence", y_true, y_prob, y_pred))
            test_set = test_set.rename(columns={kp_column[dataset]: "kp_forecast"})
            y_true, y_prob, y_pred = climatologyBase(test_set, lead_day)
            print(f"{y_true.shape}, {y_pred.shape}")
            rows.append(metricsTable(dataset, lead_day, "Climatology", y_true, y_prob, y_pred))
            #cmDisplay(y_true, y_pred)
            tables[dataset][lead_day] = pd.DataFrame(rows)

    for dataset in tables:
        for lead_day in tables[dataset]:
            tables[dataset][lead_day].to_csv(
                f"model_metrics_{dataset}_L{lead_day}.csv", index=False)

if __name__ == "__main__":
    main()