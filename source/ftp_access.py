import ftplib
import os
import sys
import time
import re
from datetime import datetime


# Docs
# Datetime: https://docs.python.org/3/library/datetime.html
# FTPLib: https://docs.python.org/3/library/ftplib.html
# Regular Expressions: https://docs.python.org/3/library/re.html#checking-for-a-pair


# Long term archive of NOAA SWPC data
FTP_HOST = "ftp.ngdc.noaa.gov"
FTP_BASE_PATH = "/STP/space-weather/swpc-products/daily_reports" # For Prediction data

DATA_PATHS  = {
    "3day": "3day_forecast",
    "geomag": "geomag_forecast",
    "daypre": "daypre"
}

def listFiles(ftp: ftplib.FTP, year: int, month: int, data_type: str):
    ''' List all files for each year and month '''
    month_str = f"{month:02d}" # To allow for leading zeros
    path = f"{FTP_BASE_PATH}/{data_type}/{year}/{month_str}"
    
    # Prevent crash on request if dir not found
    try:
        ftp.cwd(path) # Change working directory
        return ftp.nlst() # Return a list of filenames
    except ftplib.error_perm:
        print(f"Skipping missing directory: {path}")
        return []

def extractDateFile(fname):
    ''' Extract date from filename '''
    m = re.search(r"(\d{8})", fname) #  Find date in file name
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d") # Return date only
    return None

def downloadFTPfile(ftp: ftplib.FTP, year: int, month: int, fname, local_dir, data_type: str):
    ''' Single file download to local directory '''
    month_str = f"{month:02d}" # To allow for leading zeros
    remote = f"{FTP_BASE_PATH}/{data_type}/{year}/{month_str}/{fname}"
    local = os.path.join(local_dir, str(year), month_str, fname)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as f: # Write Binary (wb) to ensure data maintains format
        ftp.retrbinary(f"RETR " + remote, f.write) # Retrieve and write data
    print(f"Saved {fname} -> {local}")

def getDirBytes(path):
    # https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes

def getDirSize(size_bytes):
    ''' Get formatted directory size in string - Base_2 '''
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"
    
def showBanner(base):
    banner_name = "SPIDER_ASCII_Banner.txt"
    banner_path = os.path.join(base, "docs", banner_name)
    with open(banner_path, "r", encoding="utf-8") as f:
        print(f.read())

def downloadRange(start_date: datetime, end_date: datetime, local_dir, data_type: str):
    ''' Download range of data for specified dates from FTP Server '''
    try:
        with ftplib.FTP(FTP_HOST) as ftp:
            ftp.login() # Anonymous login
            print(f"Connected to {FTP_HOST}")
            current = start_date
            while current <= end_date:
                year, month = current.year, current.month
                files = listFiles(ftp, year, month, data_type) # List all files
                for fname in files:
                    fdate = extractDateFile(fname) # Extract date from file
                    if not fdate:
                        continue
                    if start_date <= fdate <= end_date:
                        print(f"Downloading {fname}")
                        downloadFTPfile(ftp, year, month, fname, local_dir, data_type)
                        time.sleep(0.2) # Avoid server overload
                # Jump to the first day of the next month
                if month == 12:
                    current = datetime(year + 1, 1, 1) # if December move to next year
                else:
                    current = datetime(year, month + 1, 1) # else move to next available month
        size_bytes = getDirBytes(local_dir)
        fsize = getDirSize(size_bytes) 
        print(f"Download complete. Files saved to: {local_dir}, Size: {fsize}")
        
    except ftplib.all_errors as e:
        print(f"FTP error: {e}")

def summariseDatasets(data_type:str):
    ''' Connect briefly to the FTP server and summarise available years and months. '''
    print(f"\n Checking availability for '{data_type}' on {FTP_HOST}...")
    base_path = f"{FTP_BASE_PATH}/{data_type}"
    summary = {}

    try:
        with ftplib.FTP(FTP_HOST) as ftp:
            ftp.login()
            ftp.cwd(base_path)
            years = ftp.nlst()
            for year in years:
                try:
                    ftp.cwd(f"{base_path}/{year}")
                    months = ftp.nlst()
                    summary[year] = months
                except ftplib.error_perm:
                    continue
    except ftplib.all_errors as e:
        print(f"Error retrieving summary: {e}")
        return
    
    # Display summary of data
    if summary:
        print("\nAvailable data on FTP:")
        # Build a structure to display ordered list of years and months.
        for y, months in sorted(summary.items()):
            mlist = ", ".join(months)
            print(f" {y}: {mlist}")
        print("\n")
    else:
        print("No directories found or no access.")

def userInputs():
    ''' Ask the user for data type and range and return them. '''
    choice = input("Select dataset type (3day / geomag / daypre): ").strip().lower()
    if choice not in DATA_PATHS:
        print("Invalid choice, defaulting to 'geomag'")
        choice = 'geomag'

    # Show a quick summary of what data is available
    summariseDatasets(DATA_PATHS[choice])
    
    # Allow user to insert one of the 
    start_input = input("Please enter a start date (YYYYMMDD): ").strip()
    try:
        ds = datetime.strptime(start_input, "%Y%m%d")
    except ValueError:
        print("Error: Invalid start date format. Expected YYYYMMDD.")
        sys.exit(1)

    end_input = input("Please enter an end date (YYYYMMDD): ")
    if end_input:
        try:
            de = datetime.strptime(end_input, "%Y%m%d")
        except ValueError:
            print("Error: Invalid end date format. Expected YYYYMMDD.")
            sys.exit(1)
    else:
        de = ds

    # Check date order
    if de < ds:
        print("Error: End date cannot be earlier than the start date.")
        sys.exit(1)
    
    return choice, ds, de

def runFTPaccess():
    ''' Main entry point for running ftp_access utility '''
    base = os.environ.get('SPIDER')
    if base is None:
        raise EnvironmentError("SPIDER environment variable not set!")
    
    showBanner(base)
    print("Running FTP Access Utility...")

    choice, ds, de = userInputs()

    # Format datetimes as strings for path
    start_str = ds.strftime("%Y%m%d")
    end_str = de.strftime("%Y%m%d")
    

    local_dir = os.path.join(base, "data", "raw", f"{DATA_PATHS[choice]}", f"{start_str}_{end_str}_raw")

    downloadRange(ds, de, local_dir, data_type=DATA_PATHS[choice])

if __name__ == "__main__":
    runFTPaccess()
    
    
