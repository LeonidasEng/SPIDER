import ftplib
import os
import sys
import time
import re
import logging
from datetime import datetime

def setupLogger(log_dir: str | None = None, level=logging.INFO):
    """
    Configure logger for the FTP access utility.

    The logger outputs messages to the console and, 
    if a log directory is provided stores FTP download requests 
    in a timestamped log file.

    Args:
        log_dir: Optional directory where the log file will be saved.
        level: Logging level used for console and file handler.
    
    Returns:
        logging.Logger
            Configured logger for FTP access.
    """
    logger = logging.getLogger("SPIDER.ftp")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = f"ftp_access_{timestamp}.log"
        fh = logging.FileHandler(os.path.join(log_dir, logfile))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
logger = logging.getLogger("SPIDER.ftp")

# Contains configuration for FTP hosts including base paths.
# Designed to be extended if the need arises.
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
        "tls": True # NASA SPDF requires TLS conneciton

    }
}

def listFiles(ftp: ftplib.FTP, source_cfg: dict, dataset: str, 
              year: int, month: int | None = None) -> list[str]:
    """
    List available files for a given FTP source, dataset and time period.
    
    Args:
        ftp: Active FTP object.
        source_cfg: Configuration dict for the selected FTP source.
        dataset: Dataset key used to resolve the required dataset in FTP_SOURCES.
        year: Used to construct remote path in NOAA SWPC/NGDC and NASA OMNI2.
        month: Optional value for NOAA SWPC/NGDC remote path.  

    Returns:
        list
            List of file names in target directory or empty if not target is found.
    """

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
        logger.error(f"Skipping missing directory: {path}")
        return []

def extractDateFile(fname: str):
    """
    Extract a forecast date from a NOAA SWPC filename.

    Args:
        fname (str): Filename containing a date in `YYYYMMDD` format.

    Returns:
        datetime | None:
            Parsed datetime object representing the extracted date.
            Returns `None` if no valid date pattern is found in the filename.
    """
    m = re.search(r"(19|20)\d{6}", fname) #  Find valid date in file name
    if m:
        return datetime.strptime(m.group(), "%Y%m%d") # Return date only
    return None

def extractYearFile(fname: str):
    """
    Extracts the year value from an OMNI data filename.

    Args:
        fname (str): OMNI filename expected in following format:
        `omni2_YYYY.dat`
    
    Returns:
        int | None:
            Extracted year as int. Returns `None` if filename does not match
            expected format.
    """
    m = re.match(r"^omni2_(\d{4})\.dat$", fname)
    if m:
        return int(m.group(1))
    return None

def downloadFTPfile(ftp: ftplib.FTP, source_cfg: dict, dataset: str, 
                    fname: str, local_dir: str, year: int, month: int | None = None):
    """
    Downloads a single file from an FTP source to a local directory.

    Args:
        ftp (ftplib.FTP): Active FTP connection object.
        source_cfg (dict): Source configuration dictionary containing FTP paths 
        and datasets.
        dataset (str): Dataset identifier used to resolve remote data path.
        fname (str): Name of file to download.
        year (int): Year associated with file.
        month (int | None, optional): Month associated with file.
    
    Returns: 
        None
    """
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
    
    logger.info(f"Saved {fname} -> {local}")
 
def getDirBytes(path):
    """
    Calculates the total size of a directory in bytes.

    Args:
        path (str): Path to the directory to measure.
    
    Returns:
        int:
            - Total size of all files within the directory in bytes.
    """
    # https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes

def getDirSize(size_bytes):
    """
    Converts a directory size in bytes into human-readable string.

    Args:
        size_bytes (int): Directory size in bytes.
    
    Returns:
        str: Directory size in
            - Bytes (`B`)
            - Kilobytes (`KB`)
            - Megabytes (`MB`)
            - Gigabytes (`GB`)

    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"
    
def showBanner(base:str):
    """
    Displays the SPIDER ASCII banner in the console.

    Args:
        base (str): Root SPIDER project directory containing banner file.
    
    Returns:
        None
    """
    banner_name = "SPIDER_ASCII_Banner.txt"
    banner_path = os.path.join(base, "docs", banner_name)
    with open(banner_path, "r", encoding="utf-8") as f:
        print(f.read())

def downloadRange(start_date: datetime, end_date: datetime, local_dir: str, source_key: str, dataset: str):
    """
    Downloads files within a specified date range from a configured FTP source.

    Args:
        start_date (datetime): Start date of the download range.
        end_date (datetime): End date of the download range.
        local_dir (str): Local directory to store downloaded files.
        source_key (str): Identifier for configured FTP source in `FTP_SOURCES`.
        dataset (str): Identfier used to resolve remote dataset paths.
    
    Returns:
        None
    
    Notes:
        Supports two FTP directory structures:
        - `NOAA-style`: `year/month/file`
        - `NASA-style`: `file-only` layout using yearly filenames.
        - TLS-enabled FTP connections are enabled when specified in source config.
    
    Raises:
        ftplib.all_errors:
            Logged if an FTP connection or transfer operation fails.
    """
    source_cfg = FTP_SOURCES[source_key]

    try:
        if source_cfg.get("tls", False):
            ftp = ftplib.FTP_TLS(source_cfg["host"])
            ftp.login()
            ftp.prot_p()
        else:
            ftp = ftplib.FTP(source_cfg["host"])
            ftp.login()
        logger.info(f"Connected to {source_cfg['host']} ({source_key})")
        
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
        logger.info(f"Download complete. Files saved to: {local_dir}, Size: {getDirSize(size_bytes)}")
        
    except ftplib.all_errors as e:
        logger.error(f"FTP error: {e}")
    
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except ftplib.all_errors:
                pass

def summariseDatasets(source_key: str, dataset: str):
    """
    Retrieves and displays available data from a configured FTP source
    so the user can inspect the available data and decide what date
    ranges or datasets they want to download..

    Args:
        source_key (str): Key identifying the configured FTP source from `FTP_SOURCES`.
        dataset (str): Dataset identifier used to resolve remote dataset path.

    Returns:
        None
    """
    source_cfg = FTP_SOURCES[source_key]
    dataset_path = source_cfg["datasets"].get(dataset, "")
    base_path = f"{source_cfg['base_path']}/{dataset_path}".replace("//", "/")

    logger.info(f"\n Checking availability for '{source_key} -> {dataset}'")
    logger.info(f"Host: {source_cfg['host']}")
    logger.info(f"Base path: {base_path}")

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
                logger.error("No data found.")
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
        logger.error(f"Error retrieving summary: {e}")
        return
    
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except ftplib.all_errors:
                pass

def userInputs():
    """
    Collects user inputs for FTP data retrieval

    The user is prompted to select:

    - An FTP data source.
    - A dataset within the selected source.
    - A start and end date range for download.

    Args:
        None
    
    Returns:
        tuple[str, str, datetime, datetime]:
            - source_key: Selected FTP source identifier.
            - dataset_key: Selected dataset identifier.
            - start_date: Start date for data retrieval.
            - end_date: End date for data retrieval.
    Raises:
        SystemExit: Raised if the user selects an invalid source, dataset,
        format or range.
    """
    print("Available data sources:")
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
        logger.error("Invalid start date format.")
        sys.exit(1)

    end_input = input("Enter an end date (YYYYMMDD, leave blank for same day): ").strip()
    if end_input:
        try:
            end_date = datetime.strptime(end_input, "%Y%m%d")
        except ValueError:
            logger.error("Invalid end date format.")
            sys.exit(1)
    else:
        end_date = start_date

    # Check date order
    if end_date < start_date:
        logger.error("End date cannot be earlier than the start date.")
        sys.exit(1)
    
    return source_key, dataset_key, start_date, end_date

def runFTPaccess():
    """
    Main entry point for the FTP access utility.

    The utility:
    - Displays the SPIDER startup banner.
    - Configures logging.
    - Collects user-selected FTP download parameters.
    - Creates a structure local output directory.
    - Downloads datasets within the requested date range.

    Notes:
        Downloaded datasets are stored under the SPIDER raw data directory
        using the following structure:
        `data/raw/<source>/<dataset>/<start>_<end>_raw`

        The SPIDER project root directory is resolved using the `SPIDER` environment
        variable.
    
    Raises:
        EnvironmentError: If the `SPIDER` environment variable is not defined.
    """
    base = os.environ.get('SPIDER')
    if base is None:
        raise EnvironmentError("SPIDER environment variable not set!")
    
    showBanner(base)
    logger = setupLogger(
        log_dir=os.path.join(base, "logs"),
        level=logging.INFO
    )
    logger.info("Running FTP Access Utility...")

    source_key, dataset_key, ds, de = userInputs()

    # Format datetimes as strings for path
    start_str = ds.strftime("%Y%m%d")
    end_str = de.strftime("%Y%m%d")
    
    local_dir = os.path.join(base, "data", "raw", source_key, dataset_key, f"{start_str}_{end_str}_raw")

    downloadRange(start_date=ds, end_date=de, local_dir=local_dir, source_key=source_key, dataset=dataset_key)

if __name__ == "__main__":
    runFTPaccess()
    
    
