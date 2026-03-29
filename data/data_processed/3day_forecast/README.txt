SPIDER PROCESSED DATA README
###############################################################################
The dataset parsing has gone through many different iterations and there 
have been considerable changes to the processed data outputs. 

The first version concentrated on providing a comprehensive overview of 
the NOAA 3 Day 0030 and 1230 forecasts as well as the Geomag forecasts, 
parsing all sections from the text bulletins and displaying that 
information in a machine-readable JSON format. 

There are gaps in this query (2022-2025) but it serves as a record of 
what is publicly available before NOAA provided additional data. 
This data can be found in: 
data/data_processed/3day_forecast/backup/original_query

However, due to the nature of the NCEI archive, this dataset was 
intermittent, with large data gaps where SWPC had not requested that 
the forecast be stored in the long-term NCEI archive. A contact from 
NOAA, provided all available historic data for the 0030 and 1230 
forecasts using an in-house tool.

Disclaimer: NOAA states that the data collected using this tool is 
not definitive and may still contain errors. It is a best-effort 
attempt to fill in gaps in the archive.

Forecasts began in 2011 but the products were not retained by NCEI
until 2022. This data contains all available historical 3-Day
forecasts for the period of 11-2012 to 2025, over 13 years of
data.

For the SPIDER processed dataset, 2025 was the only year without
data gaps, so the original parsing method parse_3day.py was
retained which includes geomagnetic activity forecast, solar 
radiation forecast and radio blackout forecast, including 
associated metadata. All previous years use the in-house tool.

Note: Observed and OMNI2 data are not available in this repo due to
size constraints. To reproduce the full dataset, users should run:

- `ftp_access.py`
- `parse_dayind.py`
- `parse_omni2.py`

These scripts will retrieve and process the required data for a
specified time range.
