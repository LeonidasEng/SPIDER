import os
import matplotlib as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

RANDOM_STATE = 37

TARGETS = {
        "3 Day Forecast 0030": "spider_targets_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_targets_3day_1230.parquet"
    }

def dataSplit(df: pd.DataFrame, dataset:str):
    '''
    Chronological split (not percentage split):
    Train: November 2012 -> December 2022
    Test: January 2023 -> December 2025
    '''
    df = df.sort_values(["issue_time_utc", "valid_start_utc"])

    cutoff = pd.Timestamp("2023-01-01", tz="UTC")

    train_set = df[df["issue_time_utc"] < cutoff]
    test_set = df[df["issue_time_utc"] >= cutoff]

    print(f"{dataset}")
    print(f"Train: {train_set.shape} Test: {test_set.shape}")
    return train_set, test_set

def decisionTree(train_set:pd.DataFrame, test_set:pd.DataFrame):
    
    feature_columns = [
        "bz_gsm",
        "b_mag",
        "v_sw",
        "np",
        "pdyn",
        "ey",
        "beta",
        "mach_alfven",
        "f10.7",
        "ey_int_6h",
        "bz_south_6h",
        "vsw_mean_12h",
        "vbz_coupling",
        "vbz_coupling_6h",
        "pressure_jump_flag",
        "prev_error",
        "error_rate_24h",
        "time_since_last_error"
    ] # Additional temporal features added to LR

    df_train = train_set.dropna(subset=feature_columns + ["is_large_error_win"])
    df_test = test_set.dropna(subset=feature_columns + ["is_large_error_win"])

    X_train = df_train[feature_columns]
    y_train = df_train["is_large_error_win"]

    X_test = df_test[feature_columns]
    y_test = df_test["is_large_error_win"]

    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    dt.fit(X_train, y_train)

    y_prob = dt.predict_proba(X_test)[:, 1]
    y_train_pred = dt.predict(X_train)
    y_pred = dt.predict(X_test)

    return y_test, y_train, y_train_pred, y_prob, y_pred


def cmDisplay(observed, predicted):
    
    confusion_matrix = metrics.confusion_matrix(observed, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.show()

def metricsTable(dataset, lead_day, model_name, 
                 y_test, y_prob, y_pred,
                 y_train=None, y_train_pred=None):
    
    # Persistence and Climatology don't have training sets
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    
    if y_train is not None and y_train_pred is not None:    
        train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
        fit_difference = train_accuracy - test_accuracy
        train_accuracy = round(train_accuracy, 2)
        fit_difference = round(fit_difference, 2)
    else:
        train_accuracy = None
        fit_difference = None
    
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    
    brier = metrics.brier_score_loss(y_test, y_prob) # Calibration 
    auc = metrics.roc_auc_score(y_test, y_prob) # Discrimination
    
    TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()
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
        "Train Accuracy": train_accuracy,
        "Test Accuracy": round(test_accuracy, 2),
        "Fit Difference": fit_difference,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
        "Brier": round(brier, 2),
        "AUC": round(auc, 2),
        "POD": round(pod, 2), # Same as recall, use as verification
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

    tables = {}
    
    for dataset, file_name in TARGETS.items():

        df = pd.read_parquet(os.path.join(dataset_path, file_name))
        tables[dataset] = {}

        for lead_day in [0, 1, 2]:
            rows = []

            df_ld = df[df["lead_day"] == lead_day].copy()
            df_ld = df_ld.sort_values(["issue_time_utc", "valid_start_utc"])

            # Rename before feature engineering
            df_ld = df_ld.rename(columns={"kp_threeday": "kp_forecast"})

            train_set, test_set = dataSplit(df_ld, dataset)
            print(f"Dataset: {dataset}")
            print(f"Running Decision Tree Classifier for Lead Day {lead_day}...")
            y_test, y_train, y_train_pred, y_prob, y_pred = decisionTree(train_set, test_set)
            rows.append(metricsTable(dataset, lead_day, "DT",
                                    y_test, y_prob, y_pred,
                                    y_train, y_train_pred))
    
            tables[dataset][lead_day] = pd.DataFrame(rows)

    for dataset in tables:
        for lead_day in tables[dataset]:
            tables[dataset][lead_day].to_csv(
                f"model_metrics_{dataset}_L{lead_day}.csv", index=False)
    
if __name__ == "__main__":
    main()