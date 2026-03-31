import os
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.inspection import permutation_importance

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

    train_set = addErrorFeatures(train_set)
    test_set = addErrorFeatures(test_set)

    print(f"{dataset}")
    print(f"Train: {train_set.shape} Test: {test_set.shape}")
    return train_set, test_set

def addErrorFeatures(df:pd.DataFrame):
    df = df.sort_values(["lead_day", "issue_time_utc", "valid_start_utc"])

    target = df["is_large_error_win"]

    # Adding previous error features based on persistence (derived from target pre-window)
    def _sinceLastError(series):
        # Internal and only used once for feature
        counter = 0
        output = []
        for val in series:
            if val:
                counter = 0
            else:
                counter += 1
            output.append(counter)
        return output
    
    # Previous error is inspired by the good performance of the persistence baseline model
    df["prev_error"] = df.groupby("lead_day")["is_large_error"].shift(1).fillna(False).astype(bool)

    # Attempting to avoid future leakage
    # shifted_error = df.groupby("lead_day")["is_large_error"].shift(1)

    df["error_count_24h"] = (
        df.groupby("lead_day")["is_large_error"]
        #shifted_error
        .rolling(window=8, min_periods=1) # 8 periods equal to 24-hours
        .sum()
        .reset_index(level=0, drop=True) # Reset index to original
    ) # this is not a feature and will be dropped.

    df["error_rate_24h"] = df["error_count_24h"] / 8

    df["time_since_last_error"] = (
        df.groupby("lead_day")["is_large_error"]
        # shifted_error.groupby(df["lead_day"])
        .transform(_sinceLastError)
    )

    # Need to groupby and shift at the same time
    df["error_rate_24h"] = df["error_rate_24h"].shift(1).fillna(0)
    df["time_since_last_error"] = df["time_since_last_error"].shift(1).fillna(0)

    df = df.drop("is_large_error_win", axis=1)
    df["is_large_error_win"] = target # Last column should always be the target

    df = df.sort_values(["issue_time_utc", "valid_start_utc"])

    return df

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
        min_samples_split=100,
        class_weight="balanced",
        #class_weight={0:1, 1:2},
        random_state=RANDOM_STATE
    )
    dt.fit(X_train, y_train)

    y_prob = dt.predict_proba(X_test)[:, 1]
    y_train_pred = dt.predict(X_train)
    y_pred = dt.predict(X_test)
    # y_pred = (y_prob >= 0.33).astype(int)

    return dt, X_test, y_test, y_train, y_train_pred, y_prob, y_pred

def randomForest(train_set, test_set, base, dataset, lead_day):
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

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=1
    )

    rf_cal = CalibratedClassifierCV(
        estimator=rf,
        method="sigmoid", # LR
        cv=5
    )
    rf_cal.fit(X_train, y_train)
 
    # Creating folder to store trained models (only run once)
    prefix = dataset[-4:]
    model_dir = os.path.join(base, "models")
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.join(model_dir, f"rf_cal_{prefix}_LD{lead_day}.pkl")
    joblib.dump(rf_cal, filename)

    y_prob = rf_cal.predict_proba(X_test)[:, 1]
    y_train_pred = rf_cal.predict(X_train)
    y_pred = rf_cal.predict(X_test)
    # y_pred = (y_prob >= 0.33).astype(int)

    return rf_cal, X_test, y_test, y_train, y_train_pred, y_prob, y_pred

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

def reliabilityCurve(dataset, lead_day, y_test, y_prob, model_name="Model", n_bins=10):
    """
    Generate reliability curve with probability histogram
    """

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="uniform")

    prefix = dataset[-4:]

    # Create figure with two panels in vertical configuration
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,7), gridspec_kw={"height_ratios": [2,1]})

    ax1.plot([0,1], [0,1], linestyle="--", color="grey", label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, marker="o", color="tab:blue", label=model_name)

    ax1.set_xlabel("Predicted Probability", fontsize=16)
    ax1.set_ylabel("Observed Frequency", fontsize=16)
    ax1.set_title(f"{prefix} Reliability Curve LD{lead_day} ({model_name})",
                  fontweight="bold", fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True)

    ax2.hist(y_prob, bins=n_bins, range=(0,1), edgecolor="black")
    ax2.set_xlabel("Predicted Probability", fontsize=16)
    ax2.set_ylabel("Count", fontsize=16)
    ax2.set_title("Probability Distribution", fontweight="bold", fontsize=18)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plotPFI(dataset, lead_day, model, X_test, y_test, feature_names, top=10, model_name="Model"):

    scoring_type = "roc_auc"

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring=scoring_type
    )

    importances = pd.Series(result.importances_mean, index=feature_names)

    top_features = importances.sort_values(ascending=True).tail(top)

    plt.figure(figsize=(10,6))
    top_features.plot(kind="barh")
    plt.title(f"{dataset} LD{lead_day} {model_name} Top {top} Permutation Feature Importance")
    plt.xlabel(f"{scoring_type.upper().replace('_', ' ')} Importance")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

def generateROC(y_true, y_pred, lead_day, database_name, model_name, line_colour="black"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score=y_pred)
    labels = [line.get_label() for line in plt.gca().get_lines()]
    roc_auc = metrics.auc(fpr, tpr)
    prefix = database_name[-4:]
    
    if 'Random Classifier' not in labels:
        plt.plot([0, 1], [0, 1],'--', color="grey", label="Random Classifier")

    plt.plot(fpr, tpr, color=line_colour, label=f"{model_name} %0.2f" % roc_auc)
    
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(f"{prefix} LD{lead_day} ROC Curve", fontweight="bold",fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)

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
            
            # Place test set in file for rule layer testing
            test_path = os.path.join(base, "data", "test_sets")
            os.makedirs(test_path, exist_ok=True)
            prefix = dataset[-4:]
            test_file = os.path.join(test_path, f"test_{prefix}_LD{lead_day}.parquet")
            test_set.to_parquet(test_file)

            print(f"Dataset: {dataset}")
            print(f"Running Decision Tree Classifier for Lead Day {lead_day}...")
            dt, X_test, y_test, y_train, y_train_pred, y_prob, y_pred = decisionTree(train_set, test_set)
            rows.append(metricsTable(dataset, lead_day, "DT",
                                    y_test, y_prob, y_pred,
                                    y_train, y_train_pred))
            
            # Evaluate model outputs using these tools (uncomment appropriately)
            # reliabilityCurve(dataset, lead_day, y_test, y_prob, model_name="Decision Tree")
            # cmDisplay(y_test, y_pred)
            # plotPFI(dataset, lead_day, dt, X_test, y_test, 
            #         feature_names=X_test.columns, top=10, model_name="Decision Tree")
            # generateROC(y_test, y_pred, lead_day, database_name=dataset, 
            #             model_name="DT", line_colour="tab:pink")

            print(f"Running Random Forest Classifier for Lead Day {lead_day}...")
            rf, X_test, y_test, y_train, y_train_pred, y_prob, y_pred = randomForest(train_set, test_set, 
                                                                         base, dataset, lead_day)
            rows.append(metricsTable(dataset, lead_day, "RF",
                                     y_test, y_prob, y_pred,
                                     y_train, y_train_pred))
            
            # Evaluate model outputs using these tools (uncomment appropriately)
            reliabilityCurve(dataset, lead_day, y_test, y_prob, model_name="Random Forest + LR")      
            # cmDisplay(y_test, y_pred)
            # plotPFI(dataset, lead_day, rf, X_test, y_test, 
            #         feature_names=X_test.columns, top=10, model_name="Random Forest")
            # generateROC(y_test, y_pred, lead_day, database_name=dataset, 
            #             model_name="RF", line_colour="tab:red")
    
            # # Required for all calls of ROC Curve
            # plt.show()
            tables[dataset][lead_day] = pd.DataFrame(rows)

    for dataset in tables:
        for lead_day in tables[dataset]:
            tables[dataset][lead_day].to_csv(
                f"model_metrics_{dataset}_L{lead_day}.csv", index=False)
    
if __name__ == "__main__":
    main()