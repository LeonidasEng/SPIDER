import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILES = {
        "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_features_3day_1230.parquet",
        "Geomag Forecast": "spider_features_geomag.parquet"
    }
    
def histogramObserved(dataset_path:str):
    
    bins = np.arange(-0.5, 9.6, 1)

    plt.figure(figsize=(10,6))

    obs_column = "kp_obs"

    for label, path in FILES.items():     
        df = pd.read_parquet(os.path.join(dataset_path, path))
        values = df[obs_column].dropna()

        plt.hist(values, bins=bins, alpha=0.5, label=label, edgecolor="black")

    plt.xticks(range(0,10))
    plt.xlabel("Kp Index")
    plt.ylabel("Occurences")
    plt.title("Observed Kp Distribution Comparison")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

def forecastSpread(dataset_path:str):
    df = pd.read_parquet


def loadObservedLD0(dataset_path: str, source_name: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df = df[df["lead_day"] == 0].copy() # Only Lead Day 0

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


def lineOverview(dataset_path: str):
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
    lineOverview(dataset_path)
    #linePrediction(dataset_path)

if __name__ == "__main__":
    main()