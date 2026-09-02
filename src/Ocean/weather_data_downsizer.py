import glob
import os
from pathlib import Path
import re
import datetime as dt
import time
import logging

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.Ocean.weather_utils import extract_timestamp

logging.basicConfig(
    # DEBUG < INFO < WARNING (unexpected happened, but still runs) < ERROR (function not working?) < CRITICAL (software does not work)
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# --------------------------------------------------------
# Script to subset NOAA GFS GRIB2 files to a specific lat/lon region
# and convert to NetCDF format.

# if platform.system() == "Windows":
#     INPUT_DIR = r"E:\NOAA GFS"
#     OUTPUT_DIR = r"C:\hi"

# else:
#     INPUT_DIR = "/Volumes/Weather_2/NOAA_GFS"
#     OUTPUT_DIR = "/Volumes/Weather_2/Weather_data_output"

# OUTPUT_DIR = r"C:\Users\nclemett\Documents\PhD\Data\Weather_data"
# INPUT_DIR = r"D:\NOAA GFS"
# OUTPUT_DIR = r"C:\Users\nclem\Documents\Michigan\PhD\Data\Weather_data"


VARIABLE_REGEX = (
    ":(UGRD:10 m above ground|VGRD:10 m above ground|PRMSL:mean sea level):"
)

SHORTNAMES = [
    "ws",  # Wind Speed (m/s)                        [#001]
    # "wdir",     # Wind Direction (deg)                    [#002]
    # "u",        # U-Component of Wind (m/s)               [#003]
    # "v",        # V-Component of Wind (m/s)               [#004]
    # "swh",      # Significant Height of Combined Wind Waves and Swell (m)     [#005]
    # "perpw",    # Primary Wave Mean Period (s)          [#006]
    # "dirpw",    # Primary Wave Direction (deg)          [#007]
    # "shww",     # Significant Height of Wind Waves (m)  [#008]
    "shts",  # Significant Height of Swell Waves     [#009-011]
    # "mpww",     # Mean Period of Wind Waves (s)         [#012]
    "mpts",  # Mean Period of Swell Waves            [#013-015]
    # "wvdir",    # Direction of Wind Waves (deg)          [#016]
    "swdir",  # Direction of Swell Waves (deg)        [#017-019]
]


def lat_lon_range_check(lat_s, lat_n, lon_w, lon_e):
    """
    REQUIRES: numerical values lon_w, lon_e, lat_s, lat_n
    MODIEIFES: None
    EFFECTS: Checks whether the inputs are valid
        - lat_s and lat_n within [-90, 90], and lat_s < lat_n
        - lon_w and lon_e within [-180, 360], and lon_w < lon_e
        - if not valid... raise ValueError and halts program
    """

    # latitude check
    if not (-90 <= lat_s <= 90):
        raise ValueError(f"lat_s ({lat_s}) must be between -90 and 90")
    if not (-90 <= lat_n <= 90):
        raise ValueError(f"lat_n ({lat_n}) must be between -90 and 90")

    # longitude check
    if lon_w > 180 or lon_e > 180:
        if not (0 <= lon_w <= 360 and 0 <= lon_e <= 360):
            raise ValueError("Longitude convention mismatch: both must be in [0, 360]")
    elif lon_w < 0 or lon_e < 0:
        if not (-180 <= lon_w <= 180 and -180 <= lon_e <= 180):
            raise ValueError(
                "Longitude convention mismatch: both must be in [-180, 180]"
            )
    else:
        # both values between 0 and 180 → acceptable in either convention
        pass

    # Relative position checks
    if lat_s >= lat_n:
        raise ValueError(f"lat_s ({lat_s}) must be less than lat_n ({lat_n})")
    if lon_w >= lon_e:
        raise ValueError(f"lon_w ({lon_w}) must be less than lon_e ({lon_e})")


def show_region_on_map(lat_s, lat_n, lon_w, lon_e):
    """
    REQUIRES:
        - lat_s and lat_n within [-90, 90], and lat_s < lat_n
        - lon_w and lon_e within [-180, 360], and lon_w < lon_e
    MODIFIES: None
    EFFECTS: Generates a Plate Carrée projected world map where the region
             bounded by the longitude and latitude values is highlighted
             with a red hatched rectangle. Prompts user confirmation.

    Returns:
        bool: True if user confirms region is correct (inputs 'Y' or 'y'),
              False otherwise.
    """
    # Range checks
    lat_lon_range_check(lat_s, lat_n, lon_w, lon_e)

    # Start drawing the map
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="lightgreen")
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="lightblue")
    ax.coastlines()

    rect = Rectangle(
        (lon_w, lat_s),  # bottom left corner
        width=(lon_e - lon_w),
        height=(lat_n - lat_s),
        linewidth=2,
        edgecolor="r",
        facecolor="none",
        hatch="//",
        transform=ccrs.PlateCarree(),  # Ensure alignment with map projection
    )
    ax.add_patch(rect)

    plt.title("World Map with Subset Box (red)")
    plt.show(block=False)

    confirm = (
        input("Confirm highlighted region to start downsizing--(Y/N): ").strip().lower()
    )
    return confirm.startswith("y")


def extract_date_from_filename(filename):
    """
    REQUIRES:
        filename is a string containing an 8-digit date in the format 'YYYYMMDD',
        surrounded by underscores (e.g. 'data_20241127_file.grib2')

    MODIFIES:
        None

    EFFECTS:
        Searches the given filename for a date string in 'YYYYMMDD' format
        surrounded by underscores. If found and valid, converts it to a
        datetime.datetime object and returns it.
        If no date is found or the date string is invalid, returns None.

    Returns:
        datetime.datetime or None
    """
    match = re.search(r"_([0-9]{8})_", filename)
    if not match:
        return None
    date_str = match.group(1)  # e.g. '20241127'
    try:
        return dt.datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None


def extract_forecast_hour_from_filename(filename):
    """
    REQUIRES:
        filename is a string containing a 3-digit forecast hour in the format '###',
        with an underscore before and a period after (e.g., 'data_date_369.grib2').

    MODIFIES:
        None

    EFFECTS:
        Searches the given filename for a forecast hour string as described above.
        If found and valid, converts it to an integer and returns it.
        If no valid forecast hour is found or it cannot be converted to an integer, returns None.

    RETURNS:
        int or None
    """
    match = re.search(r"_([0-9]{3})\.grib2", filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def process_single_grib(input_file, output_file, LAT_S, LAT_N, LON_W, LON_E):
    """
    REQUIRES:
        - input_file: Path to a GRIB file containing weather data.
        - output_file: Path to where the resulting NetCDF file will be saved.
        - LAT_S, LAT_N: Floats representing the southern and northern bounds of the desired subset region in degrees.
                        Must satisfy LAT_S < LAT_N, with values in [-90, 90].
        - LON_W, LON_E: Floats representing the western and eastern bounds of the desired subset region in degrees.
                        Must satisfy LON_W < LON_E.
                        Acceptable ranges:
                            - if GRIB file uses [-180, 180]: both in [-180, 180]
                            - if GRIB file uses [0, 360]: both in [0, 360]
    MODIFIES:
        - Creates a new NetCDF file at the location specified by output_file, containing the subsetted data.
    EFFECTS:
        - Opens the GRIB file at input_file.
        - Identifies the available variable shortNames.
        - Subsets the data for each desired variable listed in the global SHORTNAMES list within the specified
          lat/lon bounding box.
        - Adjusts longitude ranges if the GRIB file uses a 0-360 longitude convention.
        - Extracts and saves the subsetted variables, along with the corresponding latitude and longitude
          coordinate arrays, into a NetCDF file.
        - Logs progress and warnings to the console.

    RETURNS:
        None
    """
    logger.debug(f"Opening {input_file}")
    grbs = pygrib.open(input_file)

    # Create an empty set to collect unique shortNames
    shortnames = set()

    # Iterate over all messages and add their shortName to the set
    for grb in grbs:
        shortnames.add(grb.shortName)

    # Convert to a sorted list and log
    logger.debug(sorted(shortnames))

    # Dictionary to store the subset data for each variable
    var_data = {}
    var_units = {}
    var_levels = {}  # for swell data with multiple levels

    sample_msg = grbs.message(1)

    # Check both first and last grid point longitudes of FILE
    # 2 conventions to represent longitudes
    # 1. Set Prime Meridian as 0°, longitude WEST of PM is negative (-180° to 0°), and longitude EAST of PM is positive (0° to 180°)
    # 2. Set Prime meridian as 0°, longitude measured EASTWARD is (0° to 360°)
    lo_first = sample_msg["longitudeOfFirstGridPointInDegrees"]
    lo_last = sample_msg["longitudeOfLastGridPointInDegrees"]
    use_360 = lo_first is not None and lo_last is not None and lo_last > 180

    # If GRIB2 file uses 0 to 360 convention, make sure to change the longitude values accordingly
    if use_360:
        logger.debug("File uses 0-360 for longitudes.")
        LON_W = LON_W if LON_W >= 0 else LON_W + 360
        LON_E = LON_E if LON_E >= 0 else LON_E + 360

    # TODO: make it so that user could choose the categories wanted to get information from
    # Loop over each desired shortName and attempt to extract its data
    for sname in SHORTNAMES:
        try:
            msgs = grbs.select(shortName=sname)
            if not msgs:
                logger.warning(f"No messages found for {sname} in {input_file}.")
                continue

            # For simplicity, take the first message matching this shortName.
            #! (If the file has multiple time steps, you'll need to loop over them, like mpts, shts, and swdir)
            data_layers = []
            levels = []

            for msg in msgs:
                try:
                    data, lats, lons = msg.data(
                        lat1=LAT_S, lat2=LAT_N, lon1=LON_W, lon2=LON_E
                    )
                    if data.size == 0:
                        logger.warning(
                            f"Empty data for {sname} at level {msg.level} in {input_file}"
                        )
                        continue
                    data_layers.append(data.astype(np.float32))
                    levels.append(
                        msg.level if "level" in msg.keys() else len(data_layers) - 1
                    )
                except Exception as e:
                    logger.warning(
                        f"Error extracting {sname} level {getattr(msg, 'level', 'unknown')}: {e}"
                    )

            if len(data_layers) == 0:
                logger.warning(f"No valid data found for {sname} in {input_file}")
            elif len(data_layers) == 1:
                var_data[sname] = data_layers[0]  # shape = (lat, lon)
                logger.debug(
                    f"Extracted {sname}: shape {data_layers[0].shape} (single layer)"
                )
            else:
                var_data[sname] = np.stack(
                    data_layers, axis=0
                )  # shape = (layers, lat, lon)
                var_levels[sname] = levels
                logger.debug(
                    f"Extracted {sname}: shape {var_data[sname].shape}, levels = {levels}"
                )

            # also get the unit no matter the number of levels
            var_units[sname] = msg.units or "unknown"
            logger.debug(f"Extracted {sname}: shape {data.shape}")
        except Exception as e:
            logger.warning(
                f"Warning: Could not extract {sname} from {input_file} ({e})."
            )

    # If no data was extracted, skip file.
    if not var_data:
        logger.warning(
            f"No matching variables extracted from {input_file}. Skipping NetCDF creation."
        )
        grbs.close()
        return

    # Extract latitude and longitude arrays from one of the variables.
    # Assume that all variables have the same grid.
    sample_msg = grbs.select(shortName=SHORTNAMES[0])[0]
    _, full_lats, full_lons = sample_msg.data(
        lat1=LAT_S, lat2=LAT_N, lon1=LON_W, lon2=LON_E
    )

    grbs.close()

    # For many GRIB files on a regular grid, the latitudes and longitudes
    # can be reduced to 1D arrays. Here, we assume that the 2D lat/lon are regular.
    # For example, take the first column of latitudes and first row of longitudes.
    lat_1d = full_lats[:, 0]
    lon_1d = full_lons[0, :]

    # Create a NetCDF file and define dimensions.
    nc_out = Dataset(output_file, "w")
    nc_out.title = f"Subset of {os.path.basename(input_file)}"
    nc_out.descriptions = (
        f"Subset extracted from {os.path.basename(input_file)}.\n"
        f"Covers region from lat {LAT_S}°N to {LAT_N}°N and lon {LON_W}° to {LON_E}°.\n"
        f"Includes{shortnames}"
    )
    nc_out.lat_min = LAT_S
    nc_out.lat_max = LAT_N
    nc_out.lon_min = LON_W
    nc_out.lon_max = LON_E

    shape = var_data[SHORTNAMES[0]].shape
    if len(shape) == 2:
        nlat, nlon = shape  # (nlat, nlon)
    else:
        _, nlat, nlon = shape  # (levels, nlat, nlon), we discard the first dimension
    nc_out.createDimension("lat", nlat)
    nc_out.createDimension("lon", nlon)
    nc_out.createDimension("level", 3)

    # Create coordinate variables for latitude and longitude
    lat_nc = nc_out.createVariable("lat", "f4", ("lat",))
    lon_nc = nc_out.createVariable("lon", "f4", ("lon",))
    lat_nc.units = "degrees_north"
    lon_nc.units = "degrees_east"
    lat_nc[:] = lat_1d
    lon_nc[:] = lon_1d

    # Create a variable for each extracted shortName
    for sname, data in var_data.items():
        if data.ndim == 3:
            var_nc = nc_out.createVariable(
                sname,
                "f4",
                (
                    "level",
                    "lat",
                    "lon",
                ),
            )
        elif data.ndim == 2:
            var_nc = nc_out.createVariable(
                sname,
                "f4",
                (
                    "lat",
                    "lon",
                ),
            )
        var_nc.units = var_units.get(sname, "unknown")
        var_nc[:] = data

    # ! test the additional time variable
    valid_time = extract_timestamp(input_file)
    # Add time dimension and variable
    nc_out.createDimension("time", 1)

    time_var = nc_out.createVariable("time", "f8", ("time",))
    time_var.units = "hours since 1970-01-01 00:00:00"
    time_var.calendar = "gregorian"
    time_var.standard_name = "time"
    time_var.long_name = "forecast valid time"

    epoch = dt.datetime(1970, 1, 1)
    time_var[:] = (valid_time - epoch).total_seconds() / 3600.0

    # Optional: store as global attribute too
    nc_out.valid_time = valid_time.strftime("%Y-%m-%d %H:%M:%S")

    nc_out.close()
    logger.debug(f"Saved subset NetCDF to {output_file}")


def process_grib_directory(
    start_date_str,
    end_date_str,
    lat_s,
    lat_n,
    lon_w,
    lon_e,
    forecast_hrz,
    timestep,
    input_dir,
    output_dir,
):
    """
    Requires:
        - start_date_str and end_date_str must be in the format "%Y%m%d" (e.g., "20250101").
        - lat_s < lat_n and lon_w < lon_e (defines a valid bounding box).
        - forecast_hrz and timestep must be positive integers.
        - input_dir must be a valid path containing .grib2 files named in a consistent format
          with extractable date and forecast hour (as required by extract_date_from_filename and extract_forecast_hour_from_filename).
        - extract_date_from_filename and extract_forecast_hour_from_filename must be defined and return valid results for the files.

    Modifies:
        - The file system: may create output_dir and write .nc files into it.
        - May log warnings/info/debug messages via the logger.
        - May print progress to the console.

    Effects:
        - Processes all valid .grib2 files in input_dir that fall within the date range [start_date_str, end_date_str],
          and within the specified forecast horizon and timestep.
        - For each valid file, extracts a geographic subset and writes a corresponding .nc file to output_dir.
        - Skips files outside the date range, outside the forecast horizon, or that don't match the timestep.
        - If the bounding box is not confirmed via user interaction, exits early without processing any files.
    """

    # Convert start and end date to actual dates
    start_date = dt.datetime.strptime(start_date_str, "%Y%m%d")
    end_date = dt.datetime.strptime(end_date_str, "%Y%m%d")

    # Make sure output directory exist (if did not exist, create it anew)
    os.makedirs(output_dir, exist_ok=True)
    if timestep > forecast_hrz:
        logger.warning(
            "Warning: TIMESTEP is greater than FORECAST_HORIZON. Therefore will not be forecast data."
        )

    # 1) Confirm bounding box
    if not show_region_on_map(lat_s, lat_n, lon_w, lon_e):
        logger.warning("Region not confirmed. Exiting.")
        return
    logger.info(f"Region confirmed, processing...")
    time.sleep(0.5)

    # 2) Find all GRIB2 files in INPUT_DIR
    logger.info(f"Finding all the .grib2 files in directory {input_dir}")
    grib_files = sorted(glob.glob(os.path.join(input_dir, "*.grib2")))
    # ? Try to use a range of possible dates? so don't process everything at once? like every 4 days/

    # 3) Process each file
    total_files = len(grib_files)
    logger.info(f"processing .grib2 files")

    dot_states = [".", "..", "..."]

    for idx, grib_path in enumerate(grib_files):
        percent = ((idx + 1) / total_files) * 100
        dots = dot_states[idx % len(dot_states)]
        print(f"\rProcessing: {percent:.1f}%{dots:<3}", end="", flush=True)

        filename = os.path.basename(grib_path)

        # Check if file has valid date
        file_date = extract_date_from_filename(filename)
        if file_date is None:
            logger.warning(f"Skipping file (date not found): {filename}")
            continue

        # Check if file has forecast hour within forecast horizon
        file_forecast_hour = extract_forecast_hour_from_filename(filename)
        if file_forecast_hour is None:
            logger.warning(f"Skipping file (forecast hour not found)")
            continue
        if file_forecast_hour > forecast_hrz:
            logger.debug(f"Skipping file (out of the FORECAST HORIZON)")
            continue

        # Check if file has TIMESTEP specified
        if file_forecast_hour % timestep != 0:
            logger.debug(f"Skipping file (forecast hour is not a valid TIMESTEP)")
            continue

        if file_date >= start_date and file_date <= end_date:
            out_path = os.path.join(output_dir, filename.replace(".grib2", ".nc"))
            logger.debug(f"Processing {grib_path} -> {out_path}")
            process_single_grib(grib_path, out_path, lat_s, lat_n, lon_w, lon_e)
        else:
            logger.debug(f"Skipping file (before {start_date_str}): {filename}")

    print("\n\n")
    logger.info(f"Done! Subset files written to: {output_dir}\n")


def main():
    script_dir = Path(__file__).resolve().parent
    INPUT_DIR = script_dir.parent / "src/linked_gribs"
    input_file = os.path.join(INPUT_DIR, "*.grib2")
    OUTPUT_DIR = os.path.expanduser(
        "~/Documents/MSDL/usv_sim/src/Ocean/tests/weather_2_nc"
    )

    # With Empty LAND Values
    START_DATE = "20241218"
    END_DATE = "20241228"
    LAT_S = 30
    LAT_N = 40
    LON_W = 150
    LON_E = 160
    FORECAST_HRZ = 350
    TIMESTEP = 9

    # Extract MMDD from dates
    start_mmdd = START_DATE[4:]
    end_mmdd = END_DATE[4:]

    # Construct directory name
    dir_suffix = f"{start_mmdd}_{end_mmdd}_lat{LAT_S}_{LAT_N}_lon{LON_W}_{LON_E}_fc{FORECAST_HRZ}_ts{TIMESTEP}"
    base_dir = os.path.expanduser(script_dir.parent / "src/Ocean/tests/weather_2_nc")
    OUTPUT_DIR = os.path.join(base_dir, dir_suffix)

    # Call process_single_grib function with the input file and other parameters
    process_grib_directory(
        START_DATE,
        END_DATE,
        LAT_S,
        LAT_N,
        LON_W,
        LON_E,
        FORECAST_HRZ,
        TIMESTEP,
        INPUT_DIR,
        OUTPUT_DIR,
    )


# Standard Python pattern:
if __name__ == "__main__":
    main()
