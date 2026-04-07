import os
import glob
import datetime as dt
import netCDF4 as nc
import numpy as np
import logging


# ---------- LOGGING ----------

logger = logging.getLogger(__name__)

#! used by multi_step_plot.py and countour_map_visualizer.py and weather_data_interpolation_API.py

# ---------- FILE OPERATIONS ----------
def get_file_list(directory, pattern="*.nc"):
    """
    Requires: directory is a valid path. pattern is a valid glob pattern string (default is "*.nc").

    Modifies: None.

    Effects: Returns a sorted list of file paths in directory matching pattern. Logs the number of files found.
    """

    search_path = os.path.join(directory, pattern)
    files = sorted(glob.glob(search_path))
    logger.debug(f"{len(files)} .nc files found in {directory}")
    return files

def filelist_editor(file_list, fcst_issue_time=None):
    """
    Requires:
        - file_list: list of file paths
        - fcst_issue_time: None (nowcast) or 'YYYYMMDD_HH' string for forecast
    Modifies: Nothing
    Effects:
        - Filters files based on forecast or nowcast mode
        - Raises FileNotFoundError if no matches found
    """
    if fcst_issue_time:
        file_list = [
            f for f in file_list if f"_Global25_{fcst_issue_time}_" in os.path.basename(f)
        ]
        if not file_list:
            logger.error(f"No forecast files found issued on date: {fcst_issue_time}")
            raise FileNotFoundError(f"No forecast files found issued on date: {fcst_issue_time}")
        logger.debug(f"Filtered to {len(file_list)} forecast files for {fcst_issue_time}")
    else:
        file_list = [f for f in file_list if "_000.nc" in os.path.basename(f)]
        if not file_list:
            logger.error("No nowcast files found.")
            raise FileNotFoundError("No nowcast files found.")
        logger.debug(f"Filtered to {len(file_list)} nowcast files")

    return sorted(file_list)

# ---------- DATA UTILITIES ----------

def read_grid_info(file_path, variable_name):
    #TODO: further refactoring needed (also need to change code in contour_map_visualizer and multi-step_plot.py to incorporate get_grid_info)
    """
    Requires: 
        - file_path: path to a readable NetCDF file. 
        - variable_name: name of the variable to inspect

    Modifies: Nothing.

    Effects: 
        - Returns spatial grid info and metadata (but not actual data values).
        - ex. lats, lons, lat_min, lat_max, lon_min, lon_max, and unit
    """

    with nc.Dataset(file_path) as ds:
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        lat_min = ds.getncattr('lat_min')
        lat_max = ds.getncattr('lat_max')
        lon_min = ds.getncattr('lon_min')
        lon_max = ds.getncattr('lon_max')
        unit = ds.variables[variable_name].units
        #! what if the unit does not exist?

    logger.debug(f"Dataset bounds: lat[{lat_min}, {lat_max}], lon[{lon_min}, {lon_max}]")
    logger.debug(f"Variable '{variable_name}' has unit {unit}.")
    return lats, lons, lat_min, lat_max, lon_min, lon_max , unit

def read_variable_data(file_path, variable_name):
    """
    Requires:
        - file_path: path to a readable NetCDF file. 
        - variable_name: name of the variable to inspect

    Modifies: Nothing.

    Effects: 
        - Returns full data array for a variable (no metadata).
    """
    with nc.Dataset(file_path) as ds:
        data = ds.variables[variable_name][:]
    logger.debug(f"Read data array for '{variable_name}' from {file_path}")
    return data


def update_fcst_time(file_name):
    """
    Requires: file_name follows the format: prefix1_prefix2_YYYYMMDD_HH_FFF.nc where forecast hour is FFF.

    Modifies: None.

    Effects: Parses the file name to compute and return a tuple of (issue_datetime, forecast_datetime).
    """

    parts = file_name.split('_')
    issue_date = int(parts[2])
    issue_hr = int(parts[3])
    fcst_hr = int(parts[4].split('.')[0])
    issue_datetime = dt.datetime.strptime(f"{issue_date}{issue_hr:02d}", "%Y%m%d%H")
    fcst_datetime = issue_datetime + dt.timedelta(hours=fcst_hr)
    return issue_datetime, fcst_datetime

# ---------- HELPERS ----------
def find_nearest_index(array, value):
    """
    Requires:
        - array: 1D array-like of latitudes or longitudes
        - value: float value to search for
    Modifies: Nothing
    Effects: Returns index of the closest array value to the target
    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def extract_timestamp(filename):
    """
    Requires:
        - filename: string of format 'prefix1_prefix2_YYYYMMDD_HH_FFF.nc'
    Modifies: Nothing
    Effects: Parses filename to extract valid forecast time (init + lead hour)
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    try:
        init_time = dt.datetime.strptime(parts[2] + parts[3], "%Y%m%d%H")
        forecast_str = parts[4].split('.')[0]
        forecast_hour = int(forecast_str)
        valid_time = init_time + dt.timedelta(hours=forecast_hour)
        return valid_time
    except (IndexError, ValueError) as e:
        logger.warning(f"⚠️ Skipped: could not parse time from {filename} — {e}")
        return None    

def extract_value(file_path, variable_name, lat, lon):

    #TODO: further refactoring needed (also need to change code in multi-step_plot.py to incorporate get_grid_info)
    """
    Requires:
        - file_path: path to a NetCDF file
        - variable_name: name of the variable to extract
        - lat, lon: requested coordinates
    Modifies: Nothing.
    Effects:
        - Returns: (value, lat_used, lon_used)
        - Logs grid point used
        - Raises ValueError if grid point is too far
    """
    with nc.Dataset(file_path) as ds:
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]

        lat_idx = find_nearest_index(lats, lat)
        lon_idx = find_nearest_index(lons, lon)

        lat_actual = lats[lat_idx]
        lon_actual = lons[lon_idx]

        logger.debug(f"Using closest grid point: ({lat_actual:.3f}, {lon_actual:.3f})")

        var = ds.variables[variable_name]
        data = var[:]

        if data.ndim == 3:
            return data[0, lat_idx, lon_idx], lat_actual, lon_actual
        elif data.ndim == 2:
            return data[lat_idx, lon_idx], lat_actual, lon_actual
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
