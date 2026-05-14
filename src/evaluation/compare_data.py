import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FILES = {
        "Observed": "spider_features_obs.parquet",
        "3 Day Forecast 0030": "spider_features_3day_0030.parquet",
        "3 Day Forecast 1230": "spider_features_3day_1230.parquet",
        #"Geomag Forecast": "spider_features_geomag.parquet"
    }
    
# Helper functions for preparing datasets
def prepareForecast(df:pd.DataFrame, lead_day: int) -> pd.DataFrame:
    """
    Filters a forecast dataset to a specific forecast lead day.

    Args:
        df (pandas.DataFrame): Forecast feature dataset
        lead_day (int): Forecast lead day to filter by.
    
    Returns:
        pandas.DataFrame:
            Filtered forecast for the selected lead day. 
    """
    # Filter forecast to specific lead day
    df = df.copy()
    df = df[df["lead_day"] == lead_day]
    return df

def normaliseTime(df:pd.DataFrame) -> pd.DataFrame:
    """
    Converts valid start times to datetime format and sets as index.
    """
    # Normalise by valid time to resample Kp to daily
    df["valid_start_utc"] = pd.to_datetime(df["valid_start_utc"])
    df = df.sort_values("valid_start_utc")
    df = df.set_index("valid_start_utc")
    return df

def dailyKp(df: pd.DataFrame, kp_column: str):
    """
    Resample 3-hourly Kp data into daily max Kp values.

    Args:
        df (pandas.DataFrame): Time-indexed dataset containing 
        Kp values.
        kp_column (str): Name of Kp column to resample.
    
    Returns:
        pandas.Series:
            Daily max Kp values.
    """
    # 3-hour Kp is too noisy to plot, resampling at 1D minimum
    return df[kp_column].resample("1D").max()

def forecastRevision(dataset_path: str):
    """
    Plots forecast revision magnitude as forecast lead time decreases.

    Args:
        dataset_path (str): Path to SPIDER feature dataset directory.
    
    Returns:
        None
    
    Notes:
        This Tier 1 EDA plot compares daily max forecast Kp values across
        lead days to measure how much the forecast changes as valid 
        time approaches.

        High revision suggests new information entering forecast,
        while low revision suggests greater forecast stability.
    """

    df = pd.read_parquet(os.path.join(dataset_path, FILES["3 Day Forecast 0030"]))

    lead_days = [2, 1, 0] # Decreasing lead day

    prepared = {}
    for lead_day in lead_days:
        df_ld = prepareForecast(df, lead_day)
        df_ld = normaliseTime(df_ld)
        prepared[lead_day] = dailyKp(df_ld, "kp_threeday")

    revisions = {}

    # Step through each lead day pairing each time range
    for earlier, later in zip(lead_days[:-1], lead_days[1:]):

        df_pair = pd.concat([prepared[earlier].rename("earlier"),
                             prepared[later].rename("later")],
                             axis=1, join="inner")
        # Calculate the difference and store at new key
        revisions[f"L{earlier}_to_L{later}"] = (df_pair["earlier"] - df_pair["later"]).abs()
    
    # Plot each revision period on the figure
    for name, series in revisions.items():
        revisions[name] = series.rolling(81, center=True, min_periods=40).mean()
    
    # Plot revision for forecast
    plt.figure()
    for name, series in revisions.items():
        plt.plot(series, label=f"{name}")

    plt.ylabel("Kp Revision Magnitude", fontsize=14)
    plt.title("Forecast stability as lead time decreases (81-day rolling mean)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()


def leadDaySkill(dataset_path:str):
    """
    Plots forecast skill against lead day using absolute Kp error.

    Args:
        dataset_path (str): Path to SPIDER feature dataset directory.
    
    Returns:
        None
    
    Notes:
        Tier 2 EDA Forecast skill vs lead time. Forecast skill is measured using:

        `abs(Kp_forecast - Kp_observed)`

        Lower error indicates a more accurate forecast, while higher
        error indicates reduced forecast reliability.
    """
    lead_errors = {}
    
    kp_column = {
        "3 Day Forecast 0030": "kp_threeday",
        "3 Day Forecast 1230": "kp_threeday"
    }

    kp_obs_daily = None

    for dataset, file_name in FILES.items():
        
        df = pd.read_parquet(os.path.join(dataset_path, file_name))

        if dataset == "Observed":
            df = normaliseTime(df)
            kp_obs_daily = dailyKp(df, "kp_obs")
            continue

        if kp_obs_daily is None:
            # In case, order changes
            raise RuntimeError("Observed dataset must be loaded to identify error.")

        for lead_day in [0, 1, 2]:
            # For each lead day: filter data, sort and extract Kp
            df_ld = prepareForecast(df, lead_day)
            df_ld = normaliseTime(df_ld)
            kp_forecast = dailyKp(df_ld, kp_column[dataset]).rename("kp_forecast") # Geomag Kp different from 3 Day

            # align with observed
            aligned = pd.concat([kp_forecast, kp_obs_daily], axis=1, join="inner")
            aligned["error"] = (aligned["kp_forecast"] - aligned["kp_obs"]).abs()

            # Rolling mean to identify wider trend and de-noise
            lead_errors[lead_day] = aligned["error"].rolling(27, center=True, min_periods=10).mean()
        
        # Subplots give the best view of this data and the variation with different lead day
        fig, axs = plt.subplots(3,1, figsize=(10,6))

        colours = {
            0:"tab:blue",
            1:"tab:orange",
            2:"tab:red"
        }

        lead_order = [2, 1, 0] # Order: LD2 > LD1 > LD0 at bottom 

        for ax, ld in zip(axs, lead_order):
            ax.plot(lead_errors[ld], label=f"Lead Day {ld}", color=colours[ld])
            ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1) # Error Indicator
            ax.set_xlabel("Time", fontsize=14)
            ax.set_ylabel("Absolute Kp Error", fontsize=14)
            ax.set_ylim(0, 2.2) # Uniform range
            ax.grid(alpha=0.3)
            ax.legend()

        fig.suptitle(f"{dataset} Skill vs Lead Day", fontsize=18, fontweight="bold", y=0.95) # Move suptitle closer to plots
        fig.tight_layout()

        plt.show()

def riskCurves(dataset_path:str):
    """
    Generates operational risk curves by forecast Kp and lead day.

    Args:
        dataset_path (str): Path to the SPIDER feature dataset directory.

    Returns:
        None
    
    Notes:
        This Tier 3 EDA plot estimates the probability of a large forecast
        error on forecast Kp and lead day.
        `P((|ΔKp| > 1) | Kp, Ld)`

        The curve helps assess how trustworthu a forecast Kp value is at
        a given lead time.
    """
    # Kp columns are different across datasets
    kp_column = {
            "3 Day Forecast 0030": "kp_threeday",
            "3 Day Forecast 1230": "kp_threeday",
            #"Geomag Forecast": "kp_geomag"
        }

    for dataset, file_name in FILES.items():
        df = pd.read_parquet(os.path.join(dataset_path, file_name))

        if dataset == "Observed":
            # Define observed  and sort and set by time
            df["valid_start_utc"] = pd.to_datetime(df["valid_start_utc"])
            df = df.sort_values("valid_start_utc").set_index("valid_start_utc")
            kp_obs_daily = df["kp_obs"].resample("1D").max().rename("kp_obs")
            continue

        if kp_obs_daily is None:
            # In case, order changes
            raise RuntimeError("Observed dataset must be loaded to identify error.")

        # 1 row of 3 plots for each lead day, for each forecast
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        fig.suptitle(f"{dataset} Operational Risk by Lead Day", fontweight="bold")

        for lead_day in [0, 1, 2]:
            df_ld = prepareForecast(df, lead_day)
            df_ld = normaliseTime(df_ld)
            kp_forecast = dailyKp(df_ld, kp_column[dataset]).rename("kp_forecast")

            # Aligned dataset of each forecast by lead time with observation 
            aligned = pd.concat([kp_forecast, kp_obs_daily], axis=1, join="inner")
            aligned["error"] = (aligned["kp_forecast"] - aligned["kp_obs"]).abs()

            # Remove NaNs from aligned table and define error state
            aligned = aligned.dropna(subset=["kp_forecast", "kp_obs", "error"])
            aligned["event"] = (aligned["error"] > 1).astype(int)
            
            aligned["kp_bin"] = (aligned["kp_forecast"].round().clip(0,9)).astype(int)
            
            # Create a probability table 
            prob_table = (aligned.groupby("kp_bin")["event"] # 
                        .agg(["mean", "count"]) # specific functions: mean() and count()
                        .rename(columns={"mean":"probability"}))
            
            # Debug output table and view as percentages (percentage of error)
            # prob_table["probability"] = (prob_table["probability"] * 100).round(1).astype(str) + "%"
            # print(f"{dataset} Lead Day: {lead_day}")
            # print(prob_table)
            
            # Entries that had less than 10 counts are statistically meaningless
            prob_plot = prob_table[prob_table["count"] >= 10]
            ax = axs[lead_day]

            # Risk bands that will help identify if forecast is trustworthy
            ax.axhspan(0.0, 0.2, color="tab:green", alpha=0.2)      # Safe to trust
            ax.axhspan(0.2, 0.4, color="yellow", alpha=0.2)         # Caution
            ax.axhspan(0.4, 0.6, color="tab:orange", alpha=0.2)     # Unreliable
            ax.axhspan(0.6, 1.0, color="tab:red", alpha=0.2)        # Do not trust

            ax.plot(prob_plot.index, prob_plot["probability"], marker="o")

            ax.set_xlim(0,9) # Set x-range to Kp
            ax.set_ylim(0,1) # Set y-range to Probability
            ax.set_xlabel("Forecast Kp")
            ax.set_ylabel("P(|ΔKp| > 1)")
            display_day = lead_day + 1 if dataset == "Geomag Forecast" else lead_day
            ax.set_title(f"Lead Day {display_day}")

            ax.grid(alpha=0.3)
        
        plt.tight_layout()    
        plt.show()

def overviewObserved(dataset_path: str):
    """
    Plots obsered geomagnetic activity alongside solar radio flux.

    Args:
        dataset_path (str): Path to the SPIDER feature dataset directory.
    
    Returns:
        None
    
    Notes:
        This Tier 0 EDA plot provides long-term context for observed Kp
        across the solar cycle.

        Daily maximum Kp values are smoothed using a 27-day rolling mean,
        and storm-scale events are overlaid using NOAA G-scale categories.

        F10.7 solar radio flux is displayed on a secondary axis as a proxy
        for broader solar activity.
    """ 
    df_obs_all = pd.read_parquet(os.path.join(dataset_path, FILES["Observed"]))

    df_obs_all = normaliseTime(df_obs_all)
    df_obs_all = df_obs_all[["kp_obs", "f10.7"]]

    # With the combined set there was not a lot of new forecast records
    # Nine daily max values increased, and 1 additonal day of activity
    # the overall trend remains unchanged.

    # Find the daily max Kp, apply a 27-day rolling mean to align with solar rotation
    # https://www.sciencedirect.com/science/article/abs/pii/S027311772401086X
    kp_daily_max = df_obs_all["kp_obs"].resample("1D").max()
    kp_max_smooth = kp_daily_max.rolling(window=27, center=True, min_periods=10).mean()
    # https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-2-en.php
    f107_daily = df_obs_all["f10.7"].resample("1D").mean()

    def classifyStorm(kp):
        # These are not the finalised Kp values from GFZ-Potsdam, Germany but estimated Kp
        # from SWPC, this is intentional to maintain consitency between forecast source and
        # observed reference data when evaluating forecast reliability.
        if   kp >= 9: return "G5"
        elif kp >= 8: return "G4"
        elif kp >= 7: return "G3"
        elif kp >= 6: return "G2"
        elif kp >= 5: return "G1"
        return None # No storm
    
    # Identify storms in daily data and assign to Dataframe
    storm_points = kp_daily_max.dropna().to_frame(name="kp")
    storm_points["G"] = storm_points["kp"].apply(classifyStorm)
    storm_points = storm_points.dropna()

    # MATPLOTLIB DOCS:
    # Colours: https://matplotlib.org/stable/gallery/color/named_colors.html
    # Markers: https://matplotlib.org/stable/api/markers_api.html
    # Z Order:  https://matplotlib.org/3.1.1/gallery/misc/zorder_demo.html

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
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Kp Index", fontsize=12)
    ax1.grid(alpha=0.3)

    # Populate plot with storm scale overlay
    for g in labels.keys():
            gdata = storm_points[storm_points["G"] == g]
            ax1.scatter(gdata.index, gdata["kp"], color=colours[g], marker=markers[g], s=20, label=labels[g], zorder=zorder_map[g])
    
    # Secondary axis for F10.7
    ax2 = ax1.twinx()
    ax2.plot(f107_daily.index, f107_daily, alpha=0.35, color="magenta", label="F10.7")
    ax2.set_ylabel("F10.7 (sfu)", fontsize=12)

    handles1, labels1 = ax1.get_legend_handles_labels() # Kp and Storm scales
    handles2, labels2 = ax2.get_legend_handles_labels() # F10.7
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Geomagnetic Activity vs Solar Flux (Observed)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()    

def main():
    """
    Main entry point for SPIDER exploratory data analysis plotting.

    The pipeline:
    - Locates SPIDER feature datasets.
    - Loads selected forecast and observed datasets.
    - Runs selected plotting functions.
    - Displays comparison plots for forecast revision, forecast skill,
    operational risk or observed geomagnetic activity.

    Notes:
        This script's intended use was within development cycle so
        plotting functions need to be enabled or disabled manually
        inside main() depending on the analysis plot required.
    """
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    dataset_path = os.path.join(base, "data", "datasets")

    # Commenting out specific lines, doesn't have to be a proper program
    # I only care about making the graphs:

    #overviewObserved(dataset_path)
    forecastRevision(dataset_path)
    #leadDaySkill(dataset_path)
    #riskCurves(dataset_path)

if __name__ == "__main__":
    main()