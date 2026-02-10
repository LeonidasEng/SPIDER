import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILES = {
        "Observed": "spider_features_obs.parquet",
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
    '''
    Tier 1 EDA Disagreement between forecast products at the same
    valid time:
        - Spread = max(Kp) - min(Kp)
        - Low spread = predictable, High spread = unstable
        - Determine failures and unpredictability
    '''
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

    # Combine forecasts across datasets and compare
    df_aligned = pd.concat([
            kp_0030_ld0.rename("kp_0030"),
            kp_1230_ld0.rename("kp_1230"),
            kp_geomag_ld0.rename("kp_geomag")
        ], axis=1, join="inner")
    
    # Sanity check: alignment 
    # print("Aligned days:", len(df_aligned))
    # print(df_aligned.head())
    
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
    plt.title("Disagreement between forecast products", fontweight="bold")
    plt.legend()
    plt.show()

def forecastRevision(dataset_path:str):
    '''
    Tier 1 EDA Revision between 0030 and 1230
        - How did forecasts change between issuance
        - Add next: not just issuance but lead time (not lead day)
        - Add next: Change to iterative function using helper functions
    
    '''
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
    
    lead_errors = {}
    
    kp_column = {
        "3 Day Forecast 0030": "kp_threeday",
        "3 Day Forecast 1230": "kp_threeday",
        "Geomag Forecast": "kp_geomag"
    }

    kp_obs_daily = None

    for dataset, file_name in FILES.items():
        
        df = pd.read_parquet(os.path.join(dataset_path, file_name))

        if dataset == "Observed":
            df["valid_start_utc"] = pd.to_datetime(df["valid_start_utc"])
            df = df.sort_values("valid_start_utc").set_index("valid_start_utc")
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
            ax.set_xlabel("Time")
            ax.set_ylabel("Absolute Kp Error")
            ax.set_ylim(0, 2.2) # Uniform range
            ax.grid(alpha=0.3)
            ax.legend()

        fig.suptitle(f"{dataset} Skill vs Lead Day", fontweight="bold", y=0.95) # Move suptitle closer to plots
        fig.tight_layout()

        plt.show()

def riskCurves(dataset_path:str):
    '''
    Generates operational risk curves for each forecast and lead day.
    
    For each lead day, the function aligns daily max forecast Kp with 
    observed daily max Kp to calculate probability of a large forecast
    error P(|ΔKp| > 1) conditioned on the forecast value.

    Answers: "How trustworthy is a forecast Kp level at a given lead time?" 
            P((|ΔKp| > 1) | Kp, Ld)    
    '''
    # Kp columns are different across datasets
    kp_column = {
            "3 Day Forecast 0030": "kp_threeday",
            "3 Day Forecast 1230": "kp_threeday",
            "Geomag Forecast": "kp_geomag"
        }

    for dataset, file_name in FILES.items():
        df = pd.read_parquet(os.path.join(dataset_path, file_name))

        if dataset == "Observed":
            # Define observed 
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
            
            # Debug output table and view as percentages
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
    '''
    Provides high-level overview of Kp in the context of the solar cycle. 
        - F10.7cm provides long-term context of solar activity
        - Use lead day of 0 to get single occurrence per valid time.
        - Data gaps must be included to show trend of solar cycle
        - Identifies when storms occur and overlays the events
    '''
    
    df_obs_all = pd.read_parquet(os.path.join(dataset_path, FILES["Observed"]))

    df_obs_all["valid_start_utc"] = pd.to_datetime(df_obs_all["valid_start_utc"])
    df_obs_all = df_obs_all.set_index("valid_start_utc").sort_index()
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
        # These are not finalised Kp values from GFZ-Potsdam, Germany but estimated Kp
        # from SWPC
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
    ax2.set_ylabel("F10.7 (sfu)")

    handles1, labels1 = ax1.get_legend_handles_labels() # Kp and Storm scales
    handles2, labels2 = ax2.get_legend_handles_labels() # F10.7
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Geomagnetic Activity vs Solar Flux (Observed)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()    

def main():
    # Environment variable must be set to run this script
    base = os.environ.get("SPIDER")
    if base is None:
        raise EnvironmentError("SPIDER system variable is not set!")
    
    dataset_path = os.path.join(base, "data", "datasets")

    # Comment out specific line, doesn't have to be a proper program
    # Only care about making the graphs:

    #histogramObserved(dataset_path)
    overviewObserved(dataset_path)
    #forecastSpread(dataset_path)
    #forecastRevision(dataset_path)
    #leadDaySkill(dataset_path)
    #riskCurves(dataset_path)
    #linePrediction(dataset_path)

if __name__ == "__main__":
    main()