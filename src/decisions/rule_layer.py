import os
import sys
import pandas as pd
import joblib
import json

from datetime import datetime, timedelta

DEBUG = False

def loadModel(forecast, lead_day):
    # Path to trained models
    model_path = f"models/rf_cal_{forecast}_LD{lead_day}.pkl"
    return joblib.load(model_path)

def showBanner(base):
    banner_name = "SPIDER_ASCII_Banner.txt"
    banner_path = os.path.join(base, "docs", banner_name)
    with open(banner_path, "r", encoding="utf-8") as f:
        return f.read()

def predictProb(model, X):
    # Model target is is_large_error
    return model.predict_proba(X)[:, 1]

def confidenceInterval(prob, uncertainty):
    # Prevent invalid probabilities outside [0,1]
    lower = max(0, prob - uncertainty)
    upper = min(1, prob + uncertainty)
    return lower, upper

def makeDecision(reliability, uncertainty):
    # Based on trained model and reliability curve 
    # the following decisions can be defined
    if reliability >= 0.7 and uncertainty < 0.4: # was 0.2 and 0.3
        return "HIGH CONFIDENCE, TRUST"
    elif reliability >= 0.4 and uncertainty < 0.6: # was 0.3
        return "MODERATE CONFIDENCE, CAUTION"
    else:
        return "LOW CONFIDENCE, DO NOT TRUST"

def formatBlock(df:pd.DataFrame, rels, uncertainties, cis, decisions):
    lines = []

    header = "TIME   Kp    REL   UNC   CI            DECISION"

    for i in range(len(df)):
        # Follow similar format to original forecast
        valid = df.iloc[i]["valid_start_utc"].strftime("%H:%M")
        kp = f"{df.iloc[i]['kp_forecast']:.2f}"    # Forecast Kp
        r = f"{rels[i]:.2f}"                       # Reliability
        u = f"{uncertainties[i]:.2f}"              # Uncertainty
        ci = f"[{cis[i][0]:.2f}, {cis[i][1]:.2f}]" # Confidence Interval [lo, up]
        d = decisions[i]                           # Decision

        lines.append(f"{valid}  {kp}  {r}  {u}  {ci}  {d}")
    return header + "\n" + "\n".join(lines)

def formatResult(base, ftype, new_records, model_outputs):
    issue = new_records[0].iloc[0]["issue_time_utc"]
    issue_str = datetime.strftime(issue, "%Y %b %d %H:%M UTC")
    meta = None
    meta_text = None

    lead_dates = {
        # Adding the lead day to the issue to see in 
        # same date format as original forecast
        ld: ((issue + timedelta(days=ld)).strftime("%b %d"))
        for ld in [0, 1, 2]
    }
    blocks = {}
    for ld in [0, 1, 2]:
        outputs = model_outputs[ld]

        rels = [out["reliability"] for out in outputs]
        uncs = [out["uncertainty"] for out in outputs]
        cis  = [out["ci"] for out in outputs]        
        decs = [out["decision"] for out in outputs]
        
        blocks[ld] = formatBlock(new_records[ld],
                                 rels,
                                 uncs,
                                 cis,
                                 decs)
    if DEBUG:
        if issue.year == 2025:
            metadata_path = os.path.join(base, "data", "data_processed", "3day_forecast", 
                f"3day_{ftype}", f"{issue.year}", f"3day_{issue.year}_{issue.month:02d}.json")
            data_dt = f"{issue.year}-{issue.month:02d}-{issue.day:02d}"
            with open(metadata_path, "r") as f:
                metadata_json = json.load(f)
            
            meta = metadata_json[data_dt]["kp"]["meta"]
            meta_text = ""

    
    if meta:
        rationale = meta["rationale"]
        # Rationale is two lines, splitting after full stop
        # to maintain forecast width
        rationale_lines = [s.strip() for s in rationale.split(". ") if s]
        formatted_rationale = "\n".join(rationale_lines)

        # This metadata is only available with year 2025.
        # This year had no data gaps so used parse_3day.py
        meta_text = f"""
Greatest Observed Kp {meta["greatest_observed_kp"]}
Greatest Expected Kp {meta["greatest_expected_kp"]}
Greatest Expected Scale {meta["greatest_expected_scale"]}
Rationale: {formatted_rationale}
    """
            
    # No indent here is intentional, as it would appear in txt file otherwise.
    text = f"""{showBanner(base)}
###############################################################################
:Product: SPIDER_{issue.year}{issue.month:02d}{issue.day:02d}_3DAY_FORECAST.txt
:Issued: {issue_str}
# Created by LeonidasEng, thanks to NOAA SWPC for the data!

SPIDER Geomagnetic Activity Forecast with Reliability

{lead_dates[0]} Forecast:

{blocks[0]}

{lead_dates[1]} Forecast:

{blocks[1]}

{lead_dates[2]} Forecast:

{blocks[2]}

{meta_text}

Key:
TIME: Valid start of the 3-hour forecast time window.
REL:  Reliability, the inverse of the large error probability.
UNC:  Uncertainty, the confidence in the model prediction.
      Max uncertainty is (p = 0.5), while certain is 0 or 1.
CI:   Confidence interval, uncertainty range around reliability.

DECISION: Recommendation based on results.
    """
    return text

def userInputs():
    # This is a simple prompt to act as a quick interface
    print("Please insert a date between 20230101 and 20251231 (YYYYMMDD format).")
    issue_input = input("Insert: ").strip()
    try:
        issue = datetime.strptime(issue_input, "%Y%m%d")
    except ValueError:
        print("Invalid start date format.")
        sys.exit(1)

    print("Please enter a forecast type ['0030' or '1230'].")
    ftype = input("Insert: ").strip()
    
    if ftype not in ["0030", "1230"]:
        print("Invalid forecast type.")
        sys.exit(1)
    
    return issue, ftype

def main():
    base = os.environ.get("SPIDER") # Get SPIDER $PATH

    # NOTE: Debug with these values DATE and FORECAST TYPE
    if DEBUG:
        choice = datetime.strptime("20230101", "%Y%m%d")
        ftype = "0030"
    else:
        choice, ftype = userInputs() # Specify forecast from prompt
    
    test_sets = {
        # Load the test set from data across lead days
        ld: pd.read_parquet(
            os.path.join(base, "data", "test_sets", 
                                f"test_{ftype}_LD{ld}.parquet"))
        for ld in [0, 1, 2]
    }

    new_records = {}
    
    # To avoid future leakage, 3 day forecasts were altered
    # to not include the first element of the forecast 
    # because it's valid start was before 0030 so I need to
    # correct for that change in the output from 
    # 8,8,7 to 7,8,8

    # Get all rows for the issue time (across all lead days)
    df_all = pd.concat([
        df[df["issue_time_utc"].dt.date == choice.date()]
        for df in test_sets.values()
    ])

    for ld in [0, 1, 2]:
        target_date = (choice + timedelta(days=ld)).date()

        new_records[ld] = df_all[
            df_all["valid_start_utc"].dt.date == target_date
        ].sort_values("valid_start_utc").reset_index(drop=True)
    
    if DEBUG:
        # To view all records not just one day
        new_records = {
            ld: df.sort_values("valid_start_utc").reset_index(drop=True)
            for ld, df in test_sets.items()
        }
        
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
    
    # Loads 3 CalibratedClassifierCV(RandomForestClassifier)
    models = {
        ld: loadModel(ftype, ld) for ld in [0, 1, 2]
    }

    # Create feature set for each lead day
    new_forecasts = {
        ld: df[feature_columns] for ld, df in new_records.items()
    }

    model_outputs = {}
    debug_rows = []

    for ld in [0, 1, 2]:
        model = models[ld]
        X = new_forecasts[ld]
        if DEBUG:
            df_ld = new_records[ld] # <- DEBUG

        probs = predictProb(model, X)

        outputs = []

        for i, p in enumerate(probs):
            reliability = 1 - p
            # Inverse of probability of large error = Reliability
            uncertainty = 1 - abs(2 * p - 1)
            #Defined: p=0: certain(0), p=0.5: uncertain(1), p=1: certain(0)

            # How confident?
            lower, upper = confidenceInterval(reliability, uncertainty)
            
            # Should operators trust model?
            decision = makeDecision(reliability, uncertainty)

            outputs.append({
                "reliability": reliability,
                "uncertainty": uncertainty,
                "ci": (lower, upper),
                "decision": decision
            })

            # Want to be able to see all outputs to tune decision thresholds
            if DEBUG:
                debug_rows.append({
                    "issue_time_utc": df_ld.iloc[i]["issue_time_utc"],
                    "valid_start_utc": df_ld.iloc[i]["valid_start_utc"],
                    "lead_day": ld,
                    "kp_forecast": round(df_ld.iloc[i]["kp_forecast"], 2),
                    "reliability": round(reliability, 2),
                    "uncertainty": round(uncertainty, 2),
                    "ci_lower": round(lower, 2),
                    "ci_upper": round(upper, 2),
                    "decision": decision
                })


        model_outputs[ld] = outputs 
    # Use debugger with breakpoint to vew debug_df
    debug_df = pd.DataFrame(debug_rows)
    text = formatResult(base, ftype, new_records, model_outputs)
    output_path = os.path.join(
        base, "outputs", 
        f"SPIDER_{choice.year}{choice.month:02d}{choice.day:02d}_3DAY_FORECAST.txt"
        )
    # Output forecast + reliability to file
    with open(output_path, "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()