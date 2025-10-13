'''

What did I learn from this data forecast test?
I learned that I can easily gather data from NOAA SWPC using FTP access.
I can setup a Python virtual environment for my project.
I learned to use OS library to ensure I can use my code on laptop running Linux.

I learned to manipulate JSON files with Pandas
I learned to manipulate Dataframes with Pandas functions 


Key Result: I have a way forward for my project idea, my first phase will be to build a mini-decision tree using historic data.
    - I need overlap Kp Predicted and Kp Observed to begin ML training
    - I need to focus on building datasets before assessing accuracy


'''

import os
import datetime
import pandas as pd

# Points to SPIDER = 'C:\Users\leomh\Desktop\SpaceWeather\SPIDER'
base_folder = os.environ.get("SPIDER", "").strip()      # print("SPIDER =", repr(os.environ.get("SPIDER")))
if not base_folder:
    raise EnvironmentError("SPIDER environment variable not set or EMPTY")

data_folder = os.path.join(base_folder, "data")
data_121025 = os.path.join(data_folder, "121025_forecast_test")
noaa_kp_forecast = os.path.join(data_121025, "noaa-planetary-k-index-forecast_20251005_20251015.json")

df = pd.read_json(noaa_kp_forecast)
df.columns = df.iloc[0]
df.columns.name = None
df = df.rename(columns={'observed': 'noaa_state'})
df = df[1:].reset_index(drop=True)
df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)     # print(df['time_tag'].dt.tz) == 'UTC'
df['bin_start'] = df['time_tag'].dt.floor('3h')               # ensure bins exist before separation
df['bin_end'] = df['bin_start'] + pd.Timedelta(hours=3)

# Check Kp is numeric
df['kp'] = pd.to_numeric(df['kp'], errors='coerce')

# Separate the dataframe into three dataframes ()
# df_observed = df[df['noaa_state'] == 'observed'].reset_index(drop=True)       # Drop=True removes old index
# df_predicted = df[df['noaa_state'] == 'predicted'].reset_index(drop=True)
# df_estimated = df[~df['noaa_state'].isin(['observed', 'predicted'])].reset_index(drop=True) # Finds all known quantities and negate to find estimated rows (not in obs or pred)

# For training, drop estimated rows
df_training  = df[df['noaa_state'] != 'estimated'].reset_index(drop=True)

bins = pd.date_range(                       # use date_range to manipulate datetimes
    start=df_training['bin_start'].min(),
    end=df_training['bin_start'].max(),
    freq='3h',                              # lowercase to avoid warning
    tz='UTC'
)

# Define Main Feature with NOAA Predicted Indices
feat = (df_training[df_training['noaa_state'] == 'predicted']
        .groupby('bin_start', as_index=True)
        .agg(kp_forecast_bin=('kp', 'first'),       # 'first' = take first non-null value in column
             noaa_scale=('noaa_scale', 'first'))
        .reindex(bins))

# Observation data prepared for Label
obs = (df_training[df_training['noaa_state'] == 'observed']
       .set_index('bin_start')
       .sort_index()[['kp']]
       .rename(columns={'kp': 'kp_observed'}))

label = obs.shift(-1).rename(columns={'kp_observed': 'y_kp_next'})

dataset = feat.join(label, how='left').reset_index(names='bin_start')

# Create CSV to data collection folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(data_121025, f"dataset_{timestamp}.csv")
dataset.to_csv(out_path, index=False,
               date_format="%Y-%m-%dT%H:%M:%SZ",   # UTC 
               float_format="%.2f",               # 2dp
               na_rep="")                         # NaN representation
print("Wrote:", out_path)
