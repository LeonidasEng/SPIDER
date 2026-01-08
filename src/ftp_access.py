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

FTP_SOURCES = {
    "forecasts": {
        "host": "ftp.ngdc.noaa.gov",
        "base_path": "/STP/space-weather/swpc-products/daily_reports",
        "datasets": {
            "3day": "3day_forecast",
            "geomag": "geomag_forecast",
            "daypre": "daypre"
        }
    },
    "observed_indices": {
        "host": "ftp.ngdc.noaa.gov",
        "base_path": "/STP/space-weather/swpc-products/daily_reports/space_weather_indices",
        "datasets": {"observed": ""}
    },
    "nasa_omni":{
        "host": "spdf.gsfc.nasa.gov",
        "base_path": "/pub/data/omni/low_res_omni",
        "datasets": {"omni2": ""},
        "tls": True

    }
}

def listFiles(ftp: ftplib.FTP, source_cfg: dict, dataset: str, 
              year: int, month: int | None = None):
    ''' 
    List files for a given FTP source and dataset.
    Supports:
        - year/month layout (NOAA SWPC)
        - year layout (OMNI)
    '''

    dataset_path = source_cfg["datasets"].get(dataset, "")

    if month is not None:
        month_str = f"{month:02d}"
        path = f"{source_cfg['base_path']}/{dataset_path}/{year}/{month_str}"
    else:
        path = f"{source_cfg['base_path']}/{year}"

    path = path.replace("//", "/")

    # Prevent crash on request if dir not found
    try:
        ftp.cwd(path) # Change working directory
        return ftp.nlst() # Return a list of filenames
    except ftplib.error_perm:
        print(f"Skipping missing directory: {path}")
        return []

def extractDateFile(fname):
    ''' 
    Extract YYYYMMDD from NOAA SWPC filenames. 
    '''
    m = re.search(r"(19|20)\d{6}", fname) #  Find valid date in file name
    if m:
        return datetime.strptime(m.group(), "%Y%m%d") # Return date only
    return None

def extractYearFile(fname: str):
    '''
    Extract YYYY from OMNI filenames.
    '''
    m = re.match(r"^omni2_(\d{4})\.dat$", fname)
    if m:
        return int(m.group(1))
    return None

def downloadFTPfile(ftp: ftplib.FTP, source_cfg: dict, dataset: str, 
                    fname: str, local_dir: str, year: int, month: int | None = None):
    '''
    Download single file from FTP source to a local directory
    Supports:
        - year/month layout
        - year only layout
    '''

    dataset_path = source_cfg["datasets"].get(dataset, "")

    if month is not None:
        month_str = f"{month:02d}"
        remote = f"{source_cfg['base_path']}/{dataset_path}/{year}/{month_str}/{fname}"
        local = os.path.join(local_dir, str(year), month_str, fname)
    else:
        remote = f"{source_cfg['base_path']}/{fname}"
        local = os.path.join(local_dir, str(year), fname)

    remote = remote.replace("//", "/")
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

def downloadRange(start_date: datetime, end_date: datetime, local_dir: str, source_key: str, dataset: str):
    ''' 
    Download files for a date range from a configured FTP source 
    '''
    source_cfg = FTP_SOURCES[source_key]

    try:
        if source_cfg.get("tls", False):
            ftp = ftplib.FTP_TLS(source_cfg["host"])
            ftp.login()
            ftp.prot_p()
        else:
            ftp = ftplib.FTP(source_cfg["host"])
            ftp.login()
        print(f"Connected to {source_cfg['host']} ({source_key})")
        
        current = start_date
        
        while current <= end_date:
            year, month = current.year, current.month
            
            # OMNI-style: year only layout
            if source_key == "nasa_omni":
                # OMNI is file-only, list once
                ftp.cwd(source_cfg["base_path"])
                files = ftp.nlst()

                for fname in files:
                    fyear = extractYearFile(fname)
                    if fyear is None:
                        continue

                    if start_date.year <= fyear <= end_date.year:
                        downloadFTPfile(ftp, source_cfg, dataset, fname, local_dir, year=fyear)
                        time.sleep(0.2)
                break # IMPORTANT Do not loop
             
            # NOAA-style: year/month layout
            else:
                files = listFiles(ftp, source_cfg, dataset=dataset, year=year, month=month)
                for fname in files:
                    fdate = extractDateFile(fname)
                    if not fdate:
                        continue
                    if start_date <= fdate <= end_date:
                        downloadFTPfile(ftp, source_cfg, dataset, fname, local_dir, year=year, month=month)
                        time.sleep(0.2)

                # Jump to the first day of the next month
                if month == 12:
                    current = datetime(year + 1, 1, 1) # if December move to next year
                else:
                    current = datetime(year, month + 1, 1) # else move to next available month

        size_bytes = getDirBytes(local_dir)
        print(f"Download complete. Files saved to: {local_dir}, Size: {getDirSize(size_bytes)}")
        
    except ftplib.all_errors as e:
        print(f"FTP error: {e}")
    
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except ftplib.all_errors:
                pass

def summariseDatasets(source_key: str, dataset: str):
    ''' 
    Summarise available data for a given FTP source.
        - List available years (and months where applicable)    
    '''
    
    source_cfg = FTP_SOURCES[source_key]
    dataset_path = source_cfg["datasets"].get(dataset, "")
    base_path = f"{source_cfg['base_path']}/{dataset_path}".replace("//", "/")

    print(f"\n Checking availability for '{source_key} -> {dataset}'")
    print(f"Host: {source_cfg['host']}")
    print(f"Base path: {base_path}")

    ftp = None

    try:
        # Connect (FTP / FTPS)
        if source_cfg.get("tls", False):
            ftp = ftplib.FTP_TLS(source_cfg["host"])
            ftp.login()
            ftp.prot_p()
        else:
            ftp = ftplib.FTP(source_cfg["host"])
            ftp.login()
        
        ftp.cwd(base_path)
        entries = sorted(ftp.nlst())

        # OMNI-style handling
        if source_key == "nasa_omni":
            years = sorted(
                extractYearFile(f) for f in entries
                if extractYearFile(f) is not None
            )

            if not years:
                print("No data found.")
                return

            print("\nAvailable OMNI2 yearly data:")
            for i in range(0, len(years), 10):
                chunk = years[i:i+10]
                print(" " + ", ".join(str(y) for y in chunk))
            return

        # NOAA-style handling
        print("\nAvailable data:")
        for year in sorted(entries):
            print(f" {year}:", end="")
            
            # Year/month layout (NOAA)
            try:
                ftp.cwd(f"{base_path}/{year}")
                # Filter numeric month directories
                months = [m for m in ftp.nlst() if m.isdigit()]
                if months:
                    print(": " + ", ".join(months))
                else:
                    print(" (no sub-directories)")
            except ftplib.all_errors as e:
                print(" (year-level data)")

    except ftplib.all_errors as e:
        print(f"Error retrieving summary: {e}")
        return
    
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except ftplib.all_errors:
                pass

def userInputs():
    ''' 
    Ask the user to select:
        - FTP source
        - Dataset within FTP source
        - Start and End dates 
    '''

    print("Avaiable data sources:")
    for key in FTP_SOURCES:
        print(f" - {key}")
    
    source_key = input("Select source: ").strip()
    if source_key not in FTP_SOURCES:
        print("Invalid source selection.")
        sys.exit(1)
    
    source_cfg = FTP_SOURCES[source_key]
    
    # Select dataset
    datasets = source_cfg["datasets"]
    print("\nAvailable datasets:")
    for key in datasets:
        print(f" - {key}")
    dataset_key = input("Select dataset: ").strip()
    if dataset_key not in datasets:
        print("Invalid dataset selection.")
        sys.exit(1)

    # Data availability
    summariseDatasets(source_key, dataset_key)

    # Date range 
    start_input = input("\nEnter a start date (YYYYMMDD): ").strip()
    try:
        start_date = datetime.strptime(start_input, "%Y%m%d")
    except ValueError:
        print("Invalid start date format.")
        sys.exit(1)

    end_input = input("Enter an end date (YYYYMMDD, leave blank for same day): ").strip()
    if end_input:
        try:
            end_date = datetime.strptime(end_input, "%Y%m%d")
        except ValueError:
            print("Invalid end date format.")
            sys.exit(1)
    else:
        end_date = start_date

    # Check date order
    if end_date < start_date:
        print("End date cannot be earlier than the start date.")
        sys.exit(1)
    
    return source_key, dataset_key, start_date, end_date

def runFTPaccess():
    ''' 
    Main entry point for running ftp_access utility 
    '''
    base = os.environ.get('SPIDER')
    if base is None:
        raise EnvironmentError("SPIDER environment variable not set!")
    
    showBanner(base)
    print("Running FTP Access Utility...")

    source_key, dataset_key, ds, de = userInputs()

    # Format datetimes as strings for path
    start_str = ds.strftime("%Y%m%d")
    end_str = de.strftime("%Y%m%d")
    
    local_dir = os.path.join(base, "data", "raw", source_key, dataset_key, f"{start_str}_{end_str}_raw")

    downloadRange(start_date=ds, end_date=de, local_dir=local_dir, source_key=source_key, dataset=dataset_key)

if __name__ == "__main__":
    runFTPaccess()
    
    
