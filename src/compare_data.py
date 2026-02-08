import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILES = {
        "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_features_3day_1230.parquet",
        "Geomag Forecast": "spider_features_geomag.parquet"
    }
    
def prepareForecast(df, lead_day):
    df = df.copy()

    # Lead Day Filter
    df = df[df["lead_day"] == lead_day]

    return df

def normaliseTime(df):
    df["valid_start_utc"] = pd.to_datetime(df["valid_start_utc"])
    df = df.sort_values("valid_start_utc")
    df = df.set_index("valid_start_utc")
    return df

def dailyKp(df, kp_column):
    return df[kp_column].resample("1D").max()

def forecastSpread(dataset_path:str):
    # Load in each forecast
    df_threeday_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 0030"]))
    df_threeday_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 1230"]))
    df_geomag = pd.read_parquet(os.path.join(dataset_path, FILES["Geomag Forecast"]))

    # Apply filter to retrieve Leaad Day 0 data
    df_threeday_0030_ld0 = prepareForecast(df_threeday_0030, 0)
    df_threeday_1230_ld0 = prepareForecast(df_threeday_1230, 0)
    df_geomag_ld0 = prepareForecast(df_geomag, 0)

    # Sort by valid start time
    df_threeday_0030_ld0 = normaliseTime(df_threeday_0030_ld0)
    df_threeday_1230_ld0 = normaliseTime(df_threeday_1230_ld0)
    df_geomag_ld0 = normaliseTime(df_geomag_ld0)

    # Extract daily Kp values for each forecast
    kp_0030_ld0 = dailyKp(df_threeday_0030_ld0, "kp_threeday")
    kp_1230_ld0 = dailyKp(df_threeday_1230_ld0, "kp_threeday")
    kp_geomag_ld0 = dailyKp(df_geomag_ld0, "kp_geomag")

    df_aligned = pd.concat([
            kp_0030_ld0.rename("kp_0030"),
            kp_1230_ld0.rename("kp_1230"),
            kp_geomag_ld0.rename("kp_geomag")
        ], axis=1, join="inner")
    
    # Sanity check: alignment 
    print("Aligned days:", len(df_aligned))
    print(df_aligned.head())
    
    # Spread of forecast values
    df_aligned["spread"] = df_aligned.max(axis=1) - df_aligned.min(axis=1)
    # Short term uncertainty (spikes = forecasters unsure, dips = predictable weather)
    spread_smooth = df_aligned["spread"].rolling(27, center=True, min_periods=10).mean()
    # Long term forecast reliability over time (strongly justifies modelling!)
    smooth_long = df_aligned["spread"].rolling(81, center=True, min_periods=40).mean()

    plt.figure(figsize=(12,5))
    plt.plot(spread_smooth, label="Short-term Uncertainty")
    plt.plot(smooth_long, label="Long-term Reliability")
    plt.xlabel("Time")
    plt.ylabel("Kp Spread")
    plt.title("Disagreement between forecast products")
    plt.legend()
    plt.show()

def forecastRevision(dataset_path:str):
    # Load in each forecast
    df_threeday_0030 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 0030"]))
    df_threeday_1230 = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 1230"]))

    # Filter for Lead Day 0
    df_threeday_0030_ld0 = prepareForecast(df_threeday_0030, 0)
    df_threeday_1230_ld0 = prepareForecast(df_threeday_1230, 0)

    # Sort by valid start time
    df_threeday_0030_ld0 = normaliseTime(df_threeday_0030_ld0)
    df_threeday_1230_ld0 = normaliseTime(df_threeday_1230_ld0)

    # How do forecasts compare between issue times?
    kp_0030_ld0 = dailyKp(df_threeday_0030_ld0, "kp_threeday")
    kp_1230_ld0 = dailyKp(df_threeday_1230_ld0, "kp_threeday")

    # Combine Daily Kp values and rename to distinguish origin
    df_revision = pd.concat([kp_0030_ld0.rename("kp_0030"),kp_1230_ld0.rename("kp_1230")],
                                    axis=1, join="inner")
    
    # 
    df_revision["revision"] = (df_revision["kp_0030"] - df_revision["kp_1230"]).abs()
    rev_short = df_revision["revision"].rolling(27, center=True, min_periods=10).mean()
    rev_long = df_revision["revision"].rolling(81, center=True, min_periods=40).mean()

    plt.plot(rev_short, label="Short-term revision")
    plt.plot(rev_long, label="Long-term revision")
    plt.ylabel("Kp Revision Magnitude")
    plt.title("Forecast stability between 0030 and 1230 forecast products")
    plt.legend()
    plt.show()

def leadDaySkill(dataset_path:str):
    pass

def loadObservedLD0(dataset_path: str, source_name: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df = prepareForecast(df, 0)
    # Extract only the relevant observed data
    df["valid_start_utc"] = pd.to_datetime(df["valid_start_utc"])
    df = df[["valid_start_utc", "kp_obs", "f10.7"]]

    # Add an additional column, required for later concat
    df["source"] = source_name

    return df

def buildCombinedObs(dataset_path: str) -> pd.DataFrame:
    # Load the different datasets with a lead day of 0
    df_0030     = loadObservedLD0(os.path.join(dataset_path, FILES["3 Day Forecast 0030"]), "0030")
    df_1230     = loadObservedLD0(os.path.join(dataset_path, FILES["3 Day Forecast 1230"]), "1230")
    df_geomag   = loadObservedLD0(os.path.join(dataset_path, FILES["Geomag Forecast"]), "geomag")

    # Combine observed values into a single dataframe and order by valid start
    df_all = pd.concat([df_0030, df_1230, df_geomag], ignore_index=True)
    df_all = df_all.dropna(subset=["kp_obs"])
    df_all = df_all.sort_values("valid_start_utc")
    df_all = df_all.groupby("valid_start_utc", as_index=False).first()
    df_all = df_all.set_index("valid_start_utc").sort_index()

    return df_all


def overviewObserved(dataset_path: str):
    '''
    Provides high-level overview of Kp in the context of the solar cycle. 
        - Use lead day of 0 to get single occurrence per valid time.
        - Data gaps included to show trend of solar cycle 
    '''
    df_3day_ld0 = buildCombinedObs(dataset_path)

    # With the combined set there was not a lot of new forecast records
    # Nine daily max values increased, and 1 additonal day of activity
    # the overall trend remains unchanged.

    # Find the daily max Kp, apply a 27-day rolling mean to align with solar rotation
    # https://www.sciencedirect.com/science/article/abs/pii/S027311772401086X
    kp_daily_max = df_3day_ld0["kp_obs"].resample("1D").max()
    kp_max_smooth = kp_daily_max.rolling(window=27, center=True, min_periods=10).mean()
    # https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-2-en.php
    f107_daily = df_3day_ld0["f10.7"].resample("1D").mean()

    def classifyStorm(kp):
        if kp >= 8.67: return "G5"
        elif kp >= 7.34: return "G4"
        elif kp >= 6.34: return "G3"
        elif kp >= 5.34: return "G2"
        elif kp >= 4.67: return "G1"
        return None
    
    # Identify storms in daily data and assign to Dataframe
    storm_points = kp_daily_max.dropna().to_frame(name="kp")
    storm_points["G"] = storm_points["kp"].apply(classifyStorm)
    storm_points = storm_points.dropna()

    # Names of each scale for display in legend
    labels = {
         "G1": "Minor",
         "G2": "Moderate",
         "G3": "Strong",
         "G4": "Severe",
         "G5": "Extreme"
    }

    # Different colours for storm scales
    colours = {
        "G1": "turquoise",
        "G2": "gold",
        "G3": "orange",
        "G4": "red",
        "G5": "darkred"
    }

    # Different markers for storm scales
    markers = {
        "G1": "o",
        "G2": "s",
        "G3": "^",
        "G4": "D",
        "G5": "X"
    }

    # Ensure the stronger storm always appears on top
    zorder_map = {
         "G1": 3,
         "G2": 4,
         "G3": 5,
         "G4": 6,
         "G5": 7,
    }

    # Create multiple plots to be overlayed
    _, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(kp_max_smooth.index, kp_max_smooth, linewidth=2, color="tab:orange", label="Kp 27-day Mean")
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Kp Index", fontsize=12)
    ax1.grid(alpha=0.3)

    # Populate plot with storm scale overlay
    for g in ["G1", "G2", "G3", "G4", "G5"]:
            gdata = storm_points[storm_points["G"] == g]
            ax1.scatter(gdata.index, gdata["kp"], color=colours[g], marker=markers[g], s=20, label=labels[g], zorder=zorder_map[g])
    
    # Secondary axis for F10.7
    ax2 = ax1.twinx()
    ax2.plot(f107_daily.index, f107_daily, alpha=0.35, color="magenta", label="F10.7")
    ax2.set_ylabel("F10.7 (sfu)")

    handles1, labels1 = ax1.get_legend_handles_labels() # Kp and Storm scales
    handles2, labels2 = ax2.get_legend_handles_labels() # F10.7
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Geomagnetic Activity vs Solar Flux (Lead Day 0)", fontdict={"fontsize": 16, "fontweight": "bold"})
    plt.tight_layout()
    plt.show()    

def main():
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    dataset_path = os.path.join(base, "data", "datasets")

    #histogramObserved(dataset_path)
    #overviewObserved(dataset_path)
    #forecastSpread(dataset_path)
    forecastRevision(dataset_path)
    #linePrediction(dataset_path)

if __name__ == "__main__":
    main()