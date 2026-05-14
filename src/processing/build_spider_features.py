import os
import numpy as np
import json
from datetime import datetime, timezone
import pandas as pd

# OBSERVED DATA, 3-DAY FORECAST DATA, OMNI2 DATA are all required
# before this script can run.

# This creates the master dataset before any data splits take place
# for training or testing.

DATASETS = {
    "observed_kp":"dayind",
    "3day_forecast": "3day_forecast",
    "omni2":"omni2"
}

def importFile(file_path:str):
    """
    Imports and parses JSON file.

    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        dict | list: 
            Parsed JSON data.
    
    Raises:
        ValueError: If the JSON file cannot be found.
    """
    try:
        # Helper function for reading in JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        raise ValueError(f"JSON file not found at: {file_path}.")

def extractObservedKp(observed_json:dict):
    """
    Extracts observed Kp values from processed observed geomag data.

    Args:
        observed_json (dict): Processed observed Kp JSON data.
    
    Returns:
        list[dict]:
            - valid_start_utc: Start time of the 3-hour observation bin.
            - kp_obs: Observed Kp value.
    """
    rows = []
    
    for day, data in observed_json.items():
        bins = data.get("estimated_planetary", [])

        for bin_entry in bins:
            time_str, kp_obs = bin_entry

            # 2022-01-01 00-03UT
            date_part, hour_part = time_str.split()
            start_hour = hour_part.split("-")[0]
            
            # Datetime this measurement started from (start of 3-hour bin)
            valid_start = datetime.strptime(f"{date_part} {start_hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)

            rows.append({
                "valid_start_utc": valid_start,
                "kp_obs": kp_obs
            })
    return rows
            
def extract3dayForecastKp(three_day_json:str):
    """
    Extracts Kp forecast values from processed NOAA 3-day forecast data.
    
    Args:
        three_day_json (dict): Processed 3-day forecast JSON data.
    
    Returns:
        list[dict]: 
            List of forecast records containing issue time, valid start time,
            forecast Kp, lead day and lead time.
    
    Notes:
        Forecast rows with negative lead times are ignored because they occur
        before the forecast issue time.
    """
    rows = []

    for day, forecast in three_day_json.items():
        for key, data in forecast.items():
            if key == "issue":
                # Track issue date to compute lead day and hour
                issue_date = data
                date_part, time_part, tz_part = issue_date.split() # Date Time Timezone
                start_time = time_part.rsplit(":", 1)[0] # Extract Hours and Minutes - some issues aren't on the hour
                issue_start = datetime.strptime(f"{date_part} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                continue
            elif key != "kp":
                # Ignore for now - Reintroduce other data later in project
                # Metadata could be useful later
                continue

            forecast_bins = ["n", "n+1", "n+2"] # Iterate through forecast bins
            for bin in forecast_bins:
                n_bins = data.get(bin, [])
                for bin_entry in n_bins:
                    time_str, threeday_kp, scale_kp = bin_entry
                    date_part, hour_part = time_str.split() # Date Time
                    start_hour = hour_part.split("-")[0] # 00-03UT get first part

                    valid_start = datetime.strptime(f"{date_part} {start_hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)

                    lead = valid_start - issue_start # Starting at Day n @ 12:30 for n-n+1-n+2 there will be 8, 8, 3 bins
                    lead_hours = (lead.total_seconds() / 3600) # Total seconds / Seconds per hour

                    if lead_hours < 0:
                        continue # Omit readings before issue date 00-12UT

                    rows.append({
                        "issue_time_utc": issue_start,
                        "valid_start_utc": valid_start,
                        "kp_threeday": threeday_kp,     # Three day Prediction
                        "lead_day": lead.days,          # Days since issue, expect 0,1,2
                        "lead_time": lead_hours         # Hours since issue
                    })

    return rows

def extractOmni2(omni2_json:dict):
    """
    Extracts selected hourly OMNI2 space weather parameters.

    Args:
        omni2_json (dict): Processed OMNI2 JSON data.
    
    Returns:
        list[dict]: 
            List of hourly OMNI2 records containing valid start time and
            selected upstream solar wind and IMF parameters.
    """
    rows = []

    FIELDS = [
        "bz_gsm",
        "b_mag",
        "v_sw",
        "np",
        "pdyn",
        "ey",
        "beta",
        "mach_alfven",
        "f10.7"    
    ] # Iterate through FIELDS safer

    for day, daily_data in omni2_json.items():
        for hour, hourly_data in daily_data.items():

            valid_start = datetime.strptime(f"{day} {hour}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
            
            row = {"valid_start_utc": valid_start,
                   **{field: hourly_data.get(field, {}).get("value") for field in FIELDS}} # Unpack value's value with (**)

            rows.append(row)
    
    return rows

def build3DayForecast(three_day_forecast_path:str) -> pd.DataFrame:
    """
    Builds a DataFrame containing processed 3-day Kp forecast records.

    Args:
        three_day_forecast_path (str): Path to processed 3-day forecast JSON files.
    
    Returns:
        pandas.DataFrame: 
            DataFrame of extracted 3-day forecast records
            sorted by issue time and valid start time.
    """
    threeday_data = []

    # Sort if sub-folders in a directory at path
    years = sorted(d for d in os.listdir(three_day_forecast_path)
                   if os.path.isdir(os.path.join(three_day_forecast_path, d)))
    
    # Year sub-folders inside processed 3day_forecast
    for year in years:
        year_path = os.path.join(three_day_forecast_path, year)

        # Years are broken down into monthly files
        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue
            
            file_path = os.path.join(year_path, file_name)
            three_day_json = importFile(file_path)
            
            # Extract records from JSON and store in rows
            rows = extract3dayForecastKp(three_day_json)
            threeday_data.extend(rows) # Extend data on iteration
    
    print(f"Extracted {len(threeday_data)} 3day bins")

    threeday_data.sort(key=lambda x: x["issue_time_utc"]) # sort by issue_time_utc or valid_start_utc
    df_threeday = pd.DataFrame(threeday_data) # Create DataFrame for 3day forecat
    df_threeday = df_threeday.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True) # Primary & secondary sort, reset index

    return df_threeday

def buildObserved(observed_path:str) -> pd.DataFrame:
    """
    Builds a DataFrame containing observed Kp records

    Args:
        observed_path (str): Path to processed observed Kp JSON files.
    
    Returns:
        pandas.DataFrame: 
            DataFrame of observed Kp records sorted by valid start time.
    """
    observed_data = []
    
    # Get sorted list of year sub-directories inside observed path
    years = sorted(d for d in os.listdir(observed_path) 
                   if os.path.isdir(os.path.join(observed_path, d)))
    
    for year in years:
        year_path = os.path.join(observed_path, year)
        
        # Sort and extract files in a single year directory
        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue # If not JSON
            
            file_path = os.path.join(year_path, file_name)
            observed_json = importFile(file_path)

            rows = extractObservedKp(observed_json)
            observed_data.extend(rows)
    
    print(f"Extracted {len(observed_data)} observed bins")
    
    observed_data.sort(key=lambda x: x["valid_start_utc"]) # Sort by valid start
    df_obs = pd.DataFrame(observed_data) # Create DataFrame for observed Kp
    df_obs = df_obs.sort_values("valid_start_utc").reset_index(drop=True) # Primary & secondary sort, reset index

    return df_obs

def buildOMNI(omni_path:str) -> pd.DataFrame:
    """
    Builds a DataFrame containing processed OMNI2 hourly records.

    Args:
        omni_path (str): Path to processed OMNI2 JSON files.
    
    Returns:
        pandas.DataFrame:
            DataFrame of OMNI2 records sorted by valid start time.
    """
    omni2_data = []

    # Get sorted list of year sub-directories inside observed
    years = sorted(d for d in os.listdir(omni_path)
                   if os.path.isdir(os.path.join(omni_path, d)))
    
    for year in years:
        year_path = os.path.join(omni_path, year)

        # Sort and extract files in a single year directory
        for file_name in sorted(os.listdir(year_path)):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(year_path, file_name)
            omni2_json = importFile(file_path)

            rows = extractOmni2(omni2_json)
            omni2_data.extend(rows)
    
    print(f"Extracted {len(omni2_data)} OMNI2 bins")
    
    omni2_data.sort(key=lambda x: x["valid_start_utc"]) # Sort by valid start
    df_omni = pd.DataFrame(omni2_data) # Create DataFrame for omni2
    df_omni = df_omni.sort_values("valid_start_utc").reset_index(drop=True)

    return df_omni

def buildTable3Day(df_3day:pd.DataFrame, df_obs:pd.DataFrame, df_omni:pd.DataFrame) -> pd.DataFrame:
    """
    Builds a forecast-centric SPIDER feature table for 3-day Kp forecasts.

    Args:
        df_3day (pandas.DataFrame): DataFrame containing 3-day forecast records.
        df_obs  (pandas.DataFrame): DataFrame containing observed Kp records.
        df_omni (pandas.DataFrame): DataFrame containing OMNI2 upstream space weather records.
    
    Returns:
        pandas.DataFrame:
            Merged feature table containing forecast Kp, observed Kp, OMNI2 context,
            lead-time information. Derived temporal features are also added here.
    
    Notes:
        OMNI2 hourly data is merged using backward nearest-time merge with 3-hour tolerance.
    """
    df = df_3day.copy() # Important: Table forecast-centric NOT observation-centric

    # 3DAY (attach valid time match)
    df = df.merge(df_obs[["valid_start_utc", "kp_obs"]], # Align with valid time, merge kp_obs
                  how="left", on="valid_start_utc")
    
    # Drop rows without valid 3-day forecast
    df = df.dropna(subset=["issue_time_utc", "kp_threeday"])

    # Merge OMNI (Upstream context (backwards), tolerance = 3 hours)
    df = pd.merge_asof(df.sort_values("valid_start_utc"),
                       df_omni.sort_values("valid_start_utc"),
                       on="valid_start_utc",
                       direction="backward",
                       tolerance=pd.Timedelta("3h"))
    
    # Add temporal features
    df = addTemporalFeatures(df)

    df["lead_time"] = df["lead_time"].round(2) # Round to 2 decimal places 

    df = df[(df["lead_time"] >= 0) & (df["lead_time"] <= 72)]
    df = removeInvalidKp(df) # Removes invalid rows

    return df.sort_values(["issue_time_utc", "valid_start_utc"]).reset_index(drop=True)

def buildTableObserved(df_obs:pd.DataFrame, df_omni:pd.DataFrame) -> pd.DataFrame:
    """
    Builds an observation-centric SPIDER feature table.

    Args:
        df_obs (pandas.DataFrame): DataFrame containing observed Kp records.
        df_omni (pandas.DataFrame): DataFrame containing upstream space weather conditions.
    
    Returns:
        pandas.DataFrame:
            Merged observation table containing Kp and OMNI2 context.
    
    Notes:
        This dataset represents the full ground-truth period rather than the forecast limited
        dataset.
    """
    df = df_obs.copy() # Full ground truth.
    
    # Merge OMNI (Upstream context for full period)
    df = pd.merge_asof(df.sort_values("valid_start_utc"),
                       df_omni.sort_values("valid_start_utc"),
                       on="valid_start_utc",
                       direction="backward",
                       tolerance=pd.Timedelta("3h"))
    
    df = removeInvalidKp(df)
    
    return df.sort_values("valid_start_utc").reset_index(drop=True)

def getProcDatapath(base:str, dataset_key: str, sub_folder: str | None = None):
    """
    Builds a processed data path for a dataset.
    
    Args:
        base (str): Root SPIDER project directory.
        dataset_key (str): Dataset key defined in the global `DATASETS` mapping.
        sub_folder (str | None, optional): Optional subfolder within the processed dataset directory.

    Returns:
        str:
            Full path to the requested processed dataset directory.
    """
    # Added subfolder param for 3day 0030 and 1230
    try:
        base_path = os.path.join(base, "data", "data_processed", DATASETS[dataset_key])
        if sub_folder:
            base_path = os.path.join(base_path, sub_folder)
        return base_path

    except:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

def removeInvalidKp(df:pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing invalid negative Kp values.

    Args:
        df (pandas.DataFrame): Input DataFrame containing one or more Kp columns.

    Returns:
        pandas.DataFrame: DataFrame with rows removed where a Kp column contained
        a negative value.
    """
    # Removes impossible Kp (Kp < 0)
    kp_columns = [c for c in ["kp_obs", "kp_threeday", "kp_geomag"] if c in df.columns]

    return df[~(df[kp_columns] < 0).any(axis=1)]

def addTemporalFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived temporal and coupling features to the forecast feature table.

    Args:
        df (pandas.DataFrame): Input forecast feature DataFrame containing OMNI2 parameters.

    Returns:
        pandas.DataFrame:
            DataFrame with additional temporal feature columns.
    
    Notes:
        Added features include: rolling Ey, southward Bz duration, rolling solar wind 
        speed, vBz coupling, rolling vBz coupling and pressure jump flag.
    """
    df = df.sort_values(["issue_time_utc", "valid_start_utc"])

    # No more than these features might envoke the Curse of Dimensionality
    # if adding more. These features are taken from the original literature.

    # 6-hour Ey (2 rows)
    df["ey_int_6h"] = df["ey"].rolling(window=2, min_periods=1).sum().round(2)

    # 6-hour Southward Bz condition (2 rows)
    df["bz_south_6h"] = (df["bz_gsm"] < 0).rolling(window=2, min_periods=1).sum().round(2)

    # 12-hour Solar Wind Speed mean (4 rows)
    df["vsw_mean_12h"] = df["v_sw"].rolling(window=4, min_periods=1).mean().round(2)

    # Product of vSw and Southward Bz
    df["vbz_coupling"] = (df["v_sw"] * np.maximum(0, -df["bz_gsm"])).round(2)

    # 6-hour rolling mean of vBz
    df["vbz_coupling_6h"] = df["vbz_coupling"].rolling(window=2, min_periods=1).mean().round(2)

    # Capture storm arrival via sudden pressure jump (bool type)
    df["pressure_jump_flag"] = df["pdyn"].diff().fillna(0) > 1.0

    return df

def verifyDataQuality(df: pd.DataFrame, name:str="dataset") -> dict:
    """
    Generates a data quality report for a SPIDER feature dataset.

    Args:
        df (pandas.DataFrame): Feature dataset to inspect.
        name (str, optional): Name used to identify the dataset in the report.
    
    Returns:
        dict:
            Data quality summary containing row counts, column counts, time coverage,
            time-gap information, missing value percentages, lead-time checks and
            duplicate time checks.
    
    Notes:
        This function adheres to the CRISP-DM data understanding stage by
        summarising consistency and performing data validation checks.
    """
    # CRISP-DM Data Understanding (2.4)
    # Examine quality of data, addressing data completeness 
    # data errors, missing values

    def percent(x):
        return round(x * 100, 3)
    
    def precision(x):
        return None if pd.isna(x) else round(float(x), 2)
    
    report = {}

    report["dataset"] = name
    report["rows"] = int(len(df))
    report["columns"] = int(df.shape[1])
    
    if "valid_start_utc" in df.columns:
        df_sorted = df.sort_values("valid_start_utc")

        # Identify time range
        report["time_start"] = df_sorted["valid_start_utc"].min()
        report["time_end"] = df_sorted["valid_start_utc"].max()

        # Identify time gaps looking for differences between rows
        differences = (df_sorted["valid_start_utc"]
                       .diff()
                       .dt.total_seconds() / 3600)
        
        # Build dictionary of gaps
        report["hour_gap_distribution"] = {
            precision(k): int(v) for k, v in differences.value_counts().to_dict().items()
        }
        # Which time steps not spaced by 3 hours (normal) or 0 hours (forecast duplicates)?
        invalid = (differences.notna()) & (differences != 0) & (differences != 3)
        report["time_gaps"] = int(invalid.sum())
        report["average_time_gaps_%"] = percent(invalid.mean())
        
        # Adding data ranges to report for more detail to report
        gap_rows = df_sorted.loc[invalid].copy()
        gap_ranges = []
        for idx in gap_rows.index:
            # Find the current time
            current_time = df_sorted.loc[idx, "valid_start_utc"]
            # Track the previous index to find gaps
            prev_idx = df_sorted.index.get_loc(idx) - 1
            if prev_idx >= 0:
                previous_time = df_sorted.iloc[prev_idx]["valid_start_utc"]
                gap_hours = (current_time - previous_time).total_seconds() / 3600
                gap_ranges.append({
                    "start_gap_after": str(previous_time),
                    "resumes_at": str(current_time),
                    "gap_hours": precision(gap_hours)
                })
        report["gap_ranges"] = gap_ranges

    
    missing_values = df.isna().mean()
    report["missing_values_%"] = {
        col: percent(val) for col, val in missing_values.items() if val > 0
    }

    if "lead_time" in df.columns:
        report["negative_lead_times"] = int((df["lead_time"] < 0).sum())
        report["max_lead_time_hours"] = precision(df["lead_time"].max())
        report["min_lead_time_hours"] = precision(df["lead_time"].min())

    if "valid_start_utc" in df.columns:
        # Multiplicity expected
        dup_mask = df.duplicated(subset=["valid_start_utc", "issue_time_utc"])
        dup = dup_mask.sum()
        report["duplicate_times"] = int(dup)
        report["duplicate_times_%"] = percent(dup / len(df))

    return report

def main():
    """
    Main entry point for building SPIDER feature datasets.

    The pipeline:
    - Locates processed observed Kp, 3-day forecast and OMNI2 data.
    - Builds DataFrames for each processed data source.
    - Merges forecasts, observed and upstream OMNI2 data.
    - Creates forecast-centric 0030 and 1230 feature datasets.
    - Creates an observation-centric ground-truth dataset.
    - Generates a feature data quality report.
    - Writes final feature datasets to parquet files.

    Notes:
        This script creates the master feature datasets before any training
        or testing splits are applied.
    """
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    # Does path exist for processed data (forecasts, observed, omni)
    observed_path = getProcDatapath(base, "observed_kp")
    three_forecast_morn_path = getProcDatapath(base, "3day_forecast", "3day_0030")
    three_forecast_aft_path = getProcDatapath(base, "3day_forecast", "3day_1230")
    omni_path = getProcDatapath(base, "omni2")

    # Build DataFrames for processed data
    df_obs = buildObserved(observed_path)
    df_3day_morn = build3DayForecast(three_forecast_morn_path)
    df_3day_aft = build3DayForecast(three_forecast_aft_path)
    df_omni = buildOMNI(omni_path)

    # Merge into forecast-centric DataFrames
    spider_3day_morn = buildTable3Day(df_3day_morn, df_obs, df_omni)
    spider_3day_aft = buildTable3Day(df_3day_aft, df_obs, df_omni)
    spider_obs = buildTableObserved(df_obs, df_omni)

    data_output_path = os.path.join(base, "data", "datasets")
    os.makedirs(data_output_path, exist_ok=True)

    path_3day_morn = os.path.join(data_output_path, "spider_features_3day_0030.parquet")
    path_3day_aft = os.path.join(data_output_path, "spider_features_3day_1230.parquet")
    path_observed = os.path.join(data_output_path, "spider_features_obs.parquet")

    report_output_path = os.path.join(base, "docs")
    os.makedirs(report_output_path, exist_ok=True)

    # Create a detailed JSON report on all feature datasets 
    with open(os.path.join(report_output_path, "spider_feature_report.json"), "w") as f:
        json.dump({
            "3day_0030": verifyDataQuality(spider_3day_morn, "3day_0030"),
            "3day_1230": verifyDataQuality(spider_3day_aft, "3day_1230")
            }, f, default=str, indent=4)

    # Output merged dataframes as parquet
    spider_3day_morn.to_parquet(path_3day_morn)
    print(f"SPIDER 3 Day 0030 feature parquet was saved to: {path_3day_morn}") 
    spider_3day_aft.to_parquet(path_3day_aft)
    print(f"SPIDER 3 Day 1230 feature parquet was saved to: {path_3day_aft}")
    spider_obs.to_parquet(path_observed)
    print(f"SPIDER Observed parquet was saved to {path_observed}")

if __name__ == "__main__":
    main()



