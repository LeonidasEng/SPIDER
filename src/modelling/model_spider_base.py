import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from scipy.optimize import curve_fit
from sklearn.metrics import roc_curve

TARGETS = {
        "3 Day Forecast 0030": "spider_targets_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_targets_3day_1230.parquet"
    }

# Create Test and Train datasets: https://www.geeksforgeeks.org/python/pandas-create-test-and-train-samples-from-dataframe/
# I'll add later when I need for NB and LR

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
    shifted_error = df.groupby("lead_day")["is_large_error"].shift(1)

    df["error_count_24h"] = (
        # df.groupby("lead_day")["is_large_error"]
        shifted_error
        .rolling(window=8, min_periods=1) # 8 periods equal to 24-hours
        .sum()
        .reset_index(level=0, drop=True) # Reset index to original
    ) # this is not a feature and will be dropped.

    df["error_rate_24h"] = df["error_count_24h"] / 8

    df["time_since_last_error"] = (
        #df.groupby("lead_day")["is_large_error"]
        shifted_error.groupby(df["lead_day"])
        .transform(_sinceLastError)
    )

    # Need to groupby and shift at the same time
    df["error_rate_24h"] = df["error_rate_24h"].fillna(0)
    df["time_since_last_error"] = df.groupby("lead_day")["time_since_last_error"].fillna(0)

    # These columns are only needed for calculation and can safely be removed
    df = df.drop(columns=["is_large_error", "error_count_24h"])

    return df

def persistenceBase(df:pd.DataFrame, target_col="is_large_error_win"):
    '''
    Was forecast wrong before? Will it be wrong again? (3-hour intervals)
    '''
    df = df.copy()
    df["probability"] = df[target_col].shift(1)

     # Remove missing values
    df = df.dropna(subset=["probability"])

    y_true = df[target_col].astype(int)         # Ground Truth
    y_prob = df["probability"].astype(float)    # Deterministic 0 or 1 but treating as prob for Brier and ROC
    y_pred = (y_prob == 1).astype(int)          # Predicted binary classification

    return y_true, y_prob, y_pred


def climatologyFit(train_set:pd.DataFrame,
                   target_col="is_large_error_win", forecast_col="kp_forecast"):
    train = train_set.copy()
    train = train.dropna(subset=[forecast_col, target_col])

    # Bin Kp for Climatology to reduce noise Kp's above 7 are grouped.
    train["kp_bin"] = train[forecast_col].round().clip(0, 7).astype(int)

    # Climatology-based probabilty model P(|ΔKp| > 1)| Forecast Kp)
    # Calculate the historical averages of the Kp scale and use it
    # for the target condition.

    # Using Laplace smoothing to fix extreme probabilities
    # https://towardsdatascience.com/laplace-smoothing-in-naive-bayes-algorithm-9c237a8bdece/
    group = train.groupby("kp_bin", observed=True)[target_col]
    errors = group.sum()
    samples = group.count()

    clim_map = ((errors + 1) / (samples + 2)).to_dict() # Fix no 1.0 probabilities

    global_mean = float(train[target_col].mean())
    return clim_map, global_mean

def climatologyApply(test_set:pd.DataFrame,
                     clim_map: dict, global_mean: float, 
                     target_col="is_large_error_win", forecast_col="kp_forecast"):
    test = test_set.copy()
    test = test.dropna(subset=[forecast_col, target_col])

    test["kp_bin"] = test[forecast_col].round().clip(0, 7).astype(int)

    test["probability"] = test["kp_bin"].map(clim_map).fillna(global_mean)

    y_true = test[target_col].astype(int)
    y_prob = test["probability"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int) # baseline prediction threshold (Camporeale:2025)
    # Changing this threshold would mean optimising the baseline which is not the objective
    # It would only effect Kp 0 and 2. 
    # This baseline answers: What skill is achievable only on historical reliability?

    return y_true, y_prob, y_pred

def gaussianBase(train_set:pd.DataFrame, test_set:pd.DataFrame):
    '''
    Gaussian Naive Bayes Baseline Model
    '''
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
    ] # Additional temporal features added to NB

    df_train = train_set.dropna(subset=feature_columns + ["is_large_error_win"])
    df_test = test_set.dropna(subset=feature_columns + ["is_large_error_win"])

    X_train = df_train[feature_columns]
    y_train = df_train["is_large_error_win"]

    X_test = df_test[feature_columns]
    y_test = df_test["is_large_error_win"]

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_true = y_test
    y_prob = nb.predict_proba(X_test)[:, 1]
    y_train_pred = nb.predict(X_train)
    y_pred = nb.predict(X_test)


    return y_test, y_train, y_train_pred, y_prob, y_pred

def logisticBase(train_set:pd.DataFrame, test_set:pd.DataFrame):
    '''
    Logistic Regression Baseline model
    '''
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

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced"))
    ]) 
    lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_test)[:, 1]
    y_train_pred = lr.predict(X_train)
    y_pred = lr.predict(X_test)

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
    ax1.legend()
    ax1.grid(True)

    ax2.hist(y_prob, bins=n_bins, range=(0,1), edgecolor="black")
    ax2.set_xlabel("Predicted Probability", fontsize=16)
    ax2.set_ylabel("Count", fontsize=16)
    ax2.set_title("Probability Distribution", fontweight="bold", fontsize=18)
    ax2.grid(alpha=0.3)

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
            base_rate_windowed = train_set["is_large_error_win"].mean()
            print(f"Windowed Base Rate for Lead Day {lead_day}: {base_rate_windowed:.3f}")

            print(f"Dataset: {dataset}")
            print(f"Running Persistence baseline model for Lead Day {lead_day}...")
            y_true, y_prob, y_pred = persistenceBase(test_set)
            rows.append(metricsTable(dataset, lead_day, "Persistence", 
                                     y_true, y_prob, y_pred))
            
            # Evaluate model outputs using these tools (uncomment appropriately)
            # cmDisplay(y_true, y_pred)
            # generateROC(y_true, y_pred, lead_day, database_name=dataset, 
            #             model_name="Persistence", line_colour="tab:blue")

            print(f"Running Climatology baseline model for Lead Day {lead_day}...")
            clim_map, global_mean = climatologyFit(train_set) # Fit to train data
            y_true, y_prob, y_pred = climatologyApply(test_set, clim_map, global_mean) # Apply to test for no future leakage
            rows.append(metricsTable(dataset, lead_day, "Climatology",  
                                     y_true, y_prob, y_pred))
            
            # Evaluate model outputs using these tools (uncomment appropriately)
            # cmDisplay(y_true, y_pred)
            # generateROC(y_true, y_pred, lead_day, database_name=dataset, 
            #             model_name="Climatology", line_colour="tab:green")

            print(f"Running Naive Bayes baseline model for Lead Day {lead_day}...")
            y_test, y_train, y_train_pred, y_prob, y_pred = gaussianBase(train_set, test_set)
            rows.append(metricsTable(dataset, lead_day, "NB",  
                                     y_test, y_prob, y_pred,
                                     y_train, y_train_pred))
            
            # Evaluate model outputs using these tools (uncomment appropriately)
            # cmDisplay(y_test, y_pred)
            # reliabilityCurve(dataset, lead_day, y_test, y_prob, model_name="Gaussian NB")
            # generateROC(y_test, y_pred, lead_day, database_name=dataset, 
            #             model_name="NB", line_colour="tab:orange")
            
            print(f"Running Logistic Regression baseline model for Lead Day {lead_day}...")
            y_test, y_train, y_train_pred, y_prob, y_pred = logisticBase(train_set, test_set)
            rows.append(metricsTable(dataset, lead_day, "LR", 
                                     y_test, y_prob, y_pred,
                                     y_train, y_train_pred))   
            
            # Evaluate model outputs using these tools (uncomment appropriately)
            # cmDisplay(y_test, y_pred)
            # reliabilityCurve(dataset, lead_day, y_test, y_prob, model_name="Logistic Regression")
            # generateROC(y_test, y_pred, lead_day, database_name=dataset, 
            #             model_name="LR", line_colour="tab:purple")
    
            # # Required for all calls of ROC Curve
            # plt.show() # For ROC Curve only

            tables[dataset][lead_day] = pd.DataFrame(rows)

    for dataset in tables:
        for lead_day in tables[dataset]:
            tables[dataset][lead_day].to_csv(
                f"base_metrics_{dataset}_L{lead_day}.csv", index=False)

if __name__ == "__main__":
    main()