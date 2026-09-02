import os
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import logging
from Ocean.weather_utils import update_fcst_time, get_file_list, extract_value

# ---------- LOGGING SETUP ----------
# logging.basicConfig(
#   level=logging.info,
#   format='%(asctime)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)

_planning_cache = {}
_nowcast_cache = {}
# --- CONFIG ---
INPUT_DIR = "Ocean/tests/weather_2_nc/1201_1210_lat10_lon10_fc50_ts9"
CURRENT_DT = datetime(2024, 12, 1, 6)
FCST_DT = datetime(2024, 12, 1, 7)
TARGET_LAT = 51.3
TARGET_LON = 221.3
VARIABLE_NAME = "wdir"


# --- HELPER FUNCTIONS ---
def get_file_metadata(file_list):
    """
    Requires:
        - file_list: a list of NetCDF file paths with naming format
                     'prefix1_prefix2_YYYYMMDD_HH_FFF.nc'
    Modifies: None.
    Effects:
        - extracts issue_datetime and forecast_datetime from files name,
        - logs skipped files if parsing fails, and
        - Returns: List of tuples: (file_path, issue_datetime, forecast_datetime)
    """
    metadata = []
    for f in file_list:
        basename = os.path.basename(f)
        try:
            issue_dt, fcst_dt = update_fcst_time(basename)
            metadata.append((f, issue_dt, fcst_dt))
        except Exception as e:
            logger.warning(f"Skipping file {f}: {e}")
    return metadata


def filter_files_by_issue_time(metadata, current_dt):
    """
    Requires:
        - metadata: list of all file data with (file_path, issue_datetime, forecast_datetime)
        - current_dt: current datetime to filter
    Modifies: None.
    Effects:
        - Calculates the last 6-hour interval at or before current_dt
        - Filters metadata to only entries with issue time equal to that
        - Returns: list of matching entries
    """
    # Find the last 6-hour interval before or equal to current_dt
    last_6hr_hour = (current_dt.hour // 6) * 6
    last_issue_time = current_dt.replace(
        hour=last_6hr_hour, minute=0, second=0, microsecond=0
    )

    # If the replacement rolled the time forward (possible on day boundaries), adjust date back
    if last_6hr_hour > current_dt.hour:
        last_issue_time -= timedelta(hours=6)
    filtered = [
        entry for entry in metadata if entry[1] == last_issue_time
    ]  # entry[1] = issue_dt
    logger.debug(
        f"Filtered metadata to {len(filtered)} entries for last issue time = {last_issue_time}"
    )
    return filtered


def find_time_bounds(filtered_metadata, fcst_dt):
    """
    Requires:
        - filtered_metadata: list of (file_path, issue_datetime, forecast_datetime)
        - fcst_dt: target forecast datetime
    Modifies: None.
    Effects:
        - finds two entries that bound fcst_dt
        - Returns: (entry_before, entry_after)
        - Raises ValueError if no bounding pair found
    """
    sorted_meta = sorted(filtered_metadata, key=lambda x: x[2])  # x[2] = fcst_dt

    for entry in sorted_meta:
        if entry[2] == fcst_dt:
            logger.debug(f"Exact match found for fcst_dt={fcst_dt}")
            return entry, entry
    for i in range(len(sorted_meta) - 1):
        t1 = sorted_meta[i][2]
        t2 = sorted_meta[i + 1][2]
        if t1 <= fcst_dt <= t2:
            logger.debug(f"Found time bounds: {t1} -- {fcst_dt} -- {t2}")
            return sorted_meta[i], sorted_meta[i + 1]

    raise ValueError(
        f"⚠️ No bounding forecast times found for interpolation. "
        f"Tried to interpolate for fcst_dt={fcst_dt}, "
        f"but available times were: {[entry[2] for entry in sorted_meta]}"
    )


def get_surrounding_indices(array, value):
    """
    Requires:
        - array: sorted 1D array (lat or lon grid)
        - value: target value to locate in array
    Modifies: None.
    Effects:
        - finds indices i1, i2 such that array[i1] <= value <= array[i2]
        - Returns: (i1, i2)
        - Raises ValueError if value out of array bounds
    """
    array = np.asarray(array)
    if len(array) < 2:
        raise ValueError("Array must contain at least two elements to interpolate.")
    if value < array[0] or value > array[-1]:
        raise ValueError(
            f"Value {value} out of bounds of array [{array[0]}, {array[-1]}]"
        )

    idx = np.searchsorted(array, value)

    if idx == 0:
        return 0, 1
    elif idx == len(array) - 1:
        return len(array) - 2, len(array) - 1
    else:
        return idx - 1, idx


def extract_value(data, lat_idx, lon_idx):
    """
    Requires:
        - data: 2D array
        - lat_idx, lon_idx: indices
    Modifies: None.
    Effects:
        - Returns: value at (lat_idx, lon_idx)
    """
    value = data[lat_idx, lon_idx]
    if np.ma.is_masked(value):
        logger.warning(f"Masked value at ({lat_idx}, {lon_idx}) — using np.nan")
        return np.nan
    logger.debug(f"value at ({lat_idx}, {lon_idx}) = {value:.3f}")
    return value


def get_spatial_weights(lats, lons, target_lat, target_lon):
    """
    Requires:
        - lats and lons are sorted 1D numpy arrays of latitude and longitude values (ascending).
        - target_lat and target_lon are within the bounds of lats and lons respectively.
    Modifies:
        - None
    Effects:
        - Computes bilinear interpolation weights for the 4 surrounding grid points.
        - Returns a tuple:
            - corners: list of 4 (lat_idx, lon_idx) index pairs
            - weights: list of 4 normalized float weights summing to 1.0
    """

    i1, i2 = get_surrounding_indices(lats, target_lat)
    j1, j2 = get_surrounding_indices(lons, target_lon)

    logger.debug(
        f"lat @ lats[{i1}] = {lats[i1]:.2f} \n\t\t\t\t"
        f"lat @ lats[{i2}] = {lats[i2]:.2f} \n\t\t\t\t"
        f"lon @ lons[{j1}] = {lons[j1]:.2f} \n\t\t\t\t"
        f"lon @ lons[{j2}] = {lons[j2]:.2f}"
    )

    dlat = lats[i2] - lats[i1]
    dlon = lons[j2] - lons[j1]

    weights = []
    corners = []

    for lat_idx in [i1, i2]:
        for lon_idx in [j1, j2]:
            lat_weight = max(0.0, 1.0 - abs(lats[lat_idx] - target_lat) / dlat)
            lon_weight = max(0.0, 1.0 - abs(lons[lon_idx] - target_lon) / dlon)
            scale = lat_weight * lon_weight
            weights.append(scale)
            corners.append((lat_idx, lon_idx))

    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [0.0 for _ in weights]

    logger.debug(f"The corners are {[(int(i), int(j)) for (i, j) in corners]}")
    logger.debug(
        f"The corresponding weights are {[f'{float(weight):.3f}' for weight in weights]}"
    )
    logger.debug(f"Totalling to {sum(weights)}")
    return corners, weights


def interpolate_spatial_value(data, corners, weights):
    """
    Requires:
        - data: 2D NumPy array of values (e.g. from a NetCDF variable) indexed by (latitude, longitude).
        - corners: list of (lat_idx, lon_idx) tuples representing the four surrounding grid points.
        - weights: list of float weights corresponding to each corner (not necessarily normalized).

    Modifies:
        - Nothing.

    Effects:
        - Retrieves values from `data` at the specified corner indices.
        - Ignores masked or NaN values.
        - Computes a weighted average of the valid values, renormalizing the weights.
        - Returns the interpolated value as a float, or np.nan if all values are masked.
    """

    values = []
    valid_weights = []

    for (lat_idx, lon_idx), w in zip(corners, weights):
        val = extract_value(data, lat_idx, lon_idx)
        if not np.isnan(val):
            values.append(val * w)
            valid_weights.append(w)

    if valid_weights:
        return sum(values) / sum(valid_weights)
    else:
        logger.warning("All values masked at this location — returning np.nan")
        return np.nan


def interpolate_all_variables(
    ds, variable_names, t1, t2, lats, lons, time_scale, target_lat, target_lon
):
    """
    Interpolates variables in a combined xarray dataset along space and time.

    Requires:
        - ds: xarray.Dataset with dims 'time', 'lat', 'lon'.
        - variable_names: list of variable names to interpolate.
        - t1, t2: datetime objects corresponding to nearest time steps in 'time'.
        - lats, lons: 1D arrays of lat/lon (ascending).
        - time_scale: float [0,1] representing interpolation weight for t2.
        - target_lat, target_lon: coordinates for interpolation.

    Returns:
        - dict: {variable_name: (interpolated_value, unit)}
    """
    results = {}
    corners, weights = get_spatial_weights(lats, lons, target_lat, target_lon)

    # Convert t1/t2 to np.datetime64 for indexing
    t164, t264 = np.datetime64(t1), np.datetime64(t2)

    for var in variable_names:
        # Extract 2D slices at t1 and t2
        try:
            data_t1 = ds[var].sel(time=t164, method="nearest").values
            data_t2 = ds[var].sel(time=t264, method="nearest").values
            unit = ds[var].attrs.get("units", "")
        except KeyError:
            logger.warning(f"Variable {var} not found in dataset.")
            results[var] = (np.nan, "")
            continue

        # Spatial interpolation
        val_t1 = interpolate_spatial_value(data_t1, corners, weights)
        val_t2 = interpolate_spatial_value(data_t2, corners, weights)

        # Temporal interpolation
        if np.isnan(val_t1) and np.isnan(val_t2):
            final_val = np.nan
        elif np.isnan(val_t1):
            final_val = val_t2
        elif np.isnan(val_t2):
            final_val = val_t1
        else:
            final_val = (1 - time_scale) * val_t1 + time_scale * val_t2

        results[var] = (final_val, unit)

    return results


# --- MAIN INTERPOLATION ---
def get_variable_unit(file_path, variable_name):
    """
    Requires:
        - file_path: path to a NetCDF file
        - variable_name: a valid variable in the file
    Modifies: None
    Effects:
        - Returns unit of the variable if available, otherwise 'unknown'
    """
    with Dataset(file_path) as ds:
        return getattr(ds.variables[variable_name], "units", "unknown")


def planning_interpolation(ds, current_dt, fcst_dt, target_lat, target_lon):
    """
    Interpolates all valid variables at a given (lat, lon) and  fcst date starting
    from the current date.
    -------------------------------------------------------------------------------
     Parameters
     ------------------------------------------------------------------------------
        ds: xarray, dataset returned from load netcdf folder function
        current_dt: datetime object, the current time
        fcst_dt: datetime object, time to get forecast at
        target_lat: float, latitude to get forecast from
        target_lon: float, longitude to get forecast from
    --------------------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------
     a dictionary of {variable_name: interpolated_value}.

    """

    if target_lon < 0:
        target_lon += 360

    fcst_dt = fcst_dt.replace(minute=0, second=0, microsecond=0)
    while fcst_dt.hour % 6 != 0:
        fcst_dt = fcst_dt - timedelta(hours=1)
    key = (
        current_dt.replace(minute=0, second=0, microsecond=0),
        fcst_dt.replace(minute=0, second=0, microsecond=0),
        round(target_lat, 1),
        round(target_lon, 1),
    )

    if key in _planning_cache:
        return _planning_cache[key]

    time_var = "time" if "time" in ds.dims else "forecast_time"

    if time_var not in ds.dims or ds[time_var].size == 0:
        logger.error(
            f"Dataset has no valid time dimension ('{time_var}') after concatenation."
        )
        return {}

    all_times = ds[time_var].values
    fcst_time64 = np.datetime64(fcst_dt)

    # Handle forecast time bounds
    times_before = all_times[all_times <= fcst_time64]
    times_after = all_times[all_times >= fcst_time64]

    if len(times_before) == 0 or len(times_after) == 0:
        logger.error("Forecast time is out of dataset bounds.")
        return {}

    t1 = times_before[-1].tolist()
    t2 = times_after[0].tolist()

    time_scale = (
        0 if t2 == t1 else (fcst_dt - t1).total_seconds() / (t2 - t1).total_seconds()
    )
    logger.debug(f"t1 = {t1}, t2 = {t2}, time scale = {time_scale:.3f}")

    lats = ds["lat"].values
    lons = ds["lon"].values
    if lats[0] > lats[-1]:
        lats = lats[::-1]
    if lons[0] > lons[-1]:
        lons = lons[::-1]

    variable_names = [
        v for v in ds.data_vars if v not in ("lat", "lon", "time", "forecast_time")
    ]

    interpolation = interpolate_all_variables(
        ds, variable_names, t1, t2, lats, lons, time_scale, target_lat, target_lon
    )

    _planning_cache[key] = interpolation
    return interpolation


def nowcast_interpolation(input_dir, fcst_dt, target_lat, target_lon):
    """
    Interpolates all valid variables at a given (lat, lon) using only nowcast files (issue_time == forecast_time).

    Requires:
        - input_dir: directory containing NetCDF nowcast files
        - fcst_dt: the datetime to interpolate to
        - target_lat, target_lon: coordinates within the data grid

    Returns:
        dict: {variable_name: (interpolated_value, unit)}
    """
    if target_lon < 0:
        target_lon += 360

    logger.info(
        f"Starting nowcast interpolation at ({target_lat}, {target_lon}) for {fcst_dt}"
    )
    key = (
        input_dir,
        fcst_dt.replace(minute=0, second=0, microsecond=0),
        round(target_lat, 1),
        round(target_lon, 1),
    )
    if key in _nowcast_cache:
        return key

    file_list = get_file_list(input_dir)
    if not file_list:
        logger.error("No NetCDF files found in input directory.")
        return {}

    file_metadata = get_file_metadata(file_list)

    # Only keep nowcast files: where issue_time == forecast_time
    nowcast_metadata = [
        (f, issue, fcst) for (f, issue, fcst) in file_metadata if issue == fcst
    ]

    if not nowcast_metadata:
        logger.error("No nowcast files (issue_time == forecast_time) found.")
        return {}

    # Find two nowcasts that bound the fcst_dt
    dt_before, dt_after = find_time_bounds(nowcast_metadata, fcst_dt)

    t1, t2 = dt_before[1], dt_after[1]
    time_scale = (fcst_dt - t1).total_seconds() / (t2 - t1).total_seconds()
    logger.info(
        f"t1 = {t1},\n\t\t\t\t t2 = {t2},\n\t\t\t\t time scale for t2 = {time_scale:.3f}"
    )

    with Dataset(dt_before[0]) as ds:
        variable_names = [
            v
            for v in ds.variables
            if ds.variables[v].ndim >= 2
            and v not in ("lat", "lon", "time", "forecast_time")
        ]
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]

    if lats[0] > lats[-1]:
        lats = lats[::-1]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
    interpolation = interpolate_all_variables(
        variable_names,
        dt_before,
        dt_after,
        lats,
        lons,
        time_scale,
        target_lat,
        target_lon,
    )
    _nowcast_cache[key] = interpolation
    return interpolation


def load_netcdf_folder(folder_path):
    """
    Loads NetCDF files from a directory efficiently, adds a synthetic time dimension
    based on forecast datetimes from filenames, and returns the combined dataset
    along with all unique issue and forecast datetimes.

    - Does NOT fully load files into memory.
    - Uses filename metadata (via helper functions) to determine time coverage.
    - Returns (combined_ds, issue_datetimes, forecast_datetimes).
    """
    import xarray as xr
    import os
    import logging

    logger = logging.getLogger(__name__)

    # --- Step 0: Collect all .nc files ---
    nc_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".nc")
        ]
    )
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {folder_path}")

    # --- Step 1: Extract metadata from filenames ---
    metadata = get_file_metadata(nc_files)
    if not metadata:
        raise RuntimeError(f"Could not extract metadata from files in {folder_path}")

    issue_datetimes = sorted(set(entry[1] for entry in metadata))
    forecast_datetimes = sorted(set(entry[2] for entry in metadata))
    logger.debug(
        f"Found {len(issue_datetimes)} issue times and {len(forecast_datetimes)} forecast times."
    )

    # --- Step 2: Define preprocess function to add synthetic 'time' dim ---
    def preprocess_add_time(ds):
        file_path = ds.encoding.get("source")
        if file_path is None:
            raise RuntimeError("Cannot determine source file for dataset")

        entry = next(e for e in metadata if e[0] == file_path)
        fcst_dt = entry[2]

        if "time" not in ds.dims:
            ds = ds.expand_dims({"time": [np.datetime64(fcst_dt)]})

        return ds

    # --- Step 3: Open all datasets efficiently using preprocess ---
    try:
        combined_ds = xr.open_mfdataset(
            nc_files,
            preprocess=preprocess_add_time,
            combine="by_coords",
            mask_and_scale=False,
            engine="netcdf4",
            parallel=False,
        )
    except Exception as e:
        # Fallback to manual concatenation if open_mfdataset fails
        logger.warning(f"open_mfdataset failed ({e}), falling back to manual open.")
        datasets = []
        for f in nc_files:
            try:
                ds = xr.open_dataset(f, mask_and_scale=False)
                ds = preprocess_add_time(ds, f)
                datasets.append(ds)
            except Exception as e2:
                logger.warning(f"Skipping {f}: {e2}")
        if not datasets:
            raise RuntimeError("No valid datasets could be loaded.")
        combined_ds = xr.concat(datasets, dim="time")

    logger.debug(f"Dataset combined successfully from {len(nc_files)} files.")
    combined_ds = combined_ds.drop_duplicates("time")
    combined_ds = combined_ds.sortby("time")
    return combined_ds, issue_datetimes, forecast_datetimes


# --- MAIN DRIVER ---
def main():
    results = planning_interpolation(
        INPUT_DIR, CURRENT_DT, FCST_DT, TARGET_LAT, TARGET_LON
    )
    print(
        f"\nInterpolated values at ({TARGET_LAT}, {TARGET_LON}) @ {FCST_DT} with issued info at/before {CURRENT_DT}:"
    )

    # results = nowcast_interpolation(INPUT_DIR, FCST_DT, TARGET_LAT, TARGET_LON)
    # print(f"\nInterpolated values at ({TARGET_LAT}, {TARGET_LON}) @ {FCST_DT}:")
    for var, (val, unit) in results.items():
        if np.isnan(val):
            print(f"{var}: NaN ({unit})")
        else:
            print(f"{var}: {val:.3f} {unit}")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    main()
