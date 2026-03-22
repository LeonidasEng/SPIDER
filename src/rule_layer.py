import os
import pandas as pd
import joblib

def loadModel(forecast, lead_day):
    # Path to trained models
    model_path = f"models/rf_cal_{forecast}_LD{lead_day}.pkl"
    return joblib.load(model_path)

def predictProb(model, X):
    # Calibrated model so probability = reliability
    return model.predict_proba(X)[:, 1]

def confidenceInterval(prob, uncertainty):
    # Prevent invalid probabilities outside [0,1]
    lower = max(0, prob - uncertainty)
    upper = min(1, prob + uncertainty)
    return lower, upper

def makeDecision(prob, uncertainty):
    # Based on trained model and reliability curve 
    # the following decisions can be defined
    if prob >= 0.7 and uncertainty < 0.2:
        return "HIGH CONFIDENCE, TRUST"
    elif prob >= 0.4 and uncertainty < 0.3:
        return "MODERATE CONFIDENCE, CAUTION"
    else:
        return "LOW CONFIDENCE, DO NOT TRUST"

def main():
    base = os.environ.get("SPIDER") # Get SPIDER $PATH

    # FIXME copy forecast style
    # Change these settings for different test sets
    ftype = "0030"     # Which forecast?
    ld = 0             # Which lead day?

    # Path to chosen forecast
    fpath = os.path.join(base, "data", "test_sets", f"test_{ftype}_LD{ld}.parquet")
    test_set = pd.read_parquet(fpath)
    # Use all the feature columns to produce probabilities
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
    ]
    
    # To select a single element
    new_record = test_set.iloc[25] # FIXME Make it a day not a single record
    new_forecast = test_set.iloc[[25]][feature_columns] # Model expects 2D entry

    results = [] # FIXME: Make this a bulletin not a row.

    model = loadModel(forecast=ftype, lead_day=ld)
    probs = predictProb(model, new_forecast)
    for p in probs:
        reliability = p
        uncertainty = 1 - abs(2*p - 1) 
        #p=0: certain(0), p=0.5: uncertain(1), p=1: certain(0)

        # How confident?
        lower, upper = confidenceInterval(p, uncertainty)
        
        # Should operators trust model?
        decision = makeDecision(p, uncertainty)

        results.append({
            "Forecast Type": ftype,
            "Lead Day": ld,
            "Issue Time":  new_record["issue_time_utc"],
            "Valid Start": new_record["valid_start_utc"],
            "Forecast Kp": new_record["kp_forecast"],
            "Reliability": f"{reliability:.2f}",
            "Uncertainty": f"{uncertainty:.2f}",
            "CI": f"[{lower:.2f}, {upper:.2f}]",
            "Decision": decision
        })
    
    final_output = pd.DataFrame(results)
    print(final_output)

if __name__ == "__main__":
    main()