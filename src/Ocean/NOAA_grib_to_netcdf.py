from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num
import numpy as np
import pygrib
import requests
import os


def convert_wind(wind_u_comp, wind_v_comp):
    """
    Convert wind from component form to magnitude and direction form.

    Parameters
    ----------
    :param wind_u_comp: 2D array of float
        gridded west to east components of wind speed
    :param wind_v_comp: 2D array of float
        gridded south to north components of wind speed

    Returns
    -------
    wind_speed : 2D array of float
        gridded magnitudes of wind speed
    wind_dir : 2D array of float
        gridded directions of wind in degrees true
    """
    # calculate speed using Euclidean distance formula
    wind_speed = np.sqrt(wind_u_comp**2 + wind_v_comp**2)
    # calculate destination direction in degrees relative
    # to the positive x-axis in a Cartesian plane (counterclockwise)
    math_deg_towards = np.degrees(np.arctan2(wind_v_comp, wind_u_comp))
    # convert from destination direction to source direction
    math_deg_from = (math_deg_towards + 180) % 360
    # convert from counterclockwise measure relative to the
    # positive x-axis to clockwise measure relative to true north
    wind_dir = (450 - math_deg_from) % 360
    return wind_speed, wind_dir


def extract_grid(lat_min, lat_max, lon_min, lon_max, grid):
    """
    Given a min/max value for lat and lon and a grib message, return
    the data grid, latitudes grid, and longitudes grid

    Parameters
    ----------
    lat_min : float
        minimum latitude value for grid
    lat_max : float
        maximum latitude value for grid
    lon_min : float
        minimum longitude value for grid
    lon_max : float
        maximum longitude value for grid
    grid : pygrib._pygrib.gribmessage
        grid object containing data to be extracted

    Returns
    -------
    full_data : numpy.ndarray
        2D array of values for variable of interest for extracted grid
    full_lats : numpy.ndarray
        2D array of latitude values for extracted grid
    full_lons : numpy.ndarray
        2D array of longitude values for extracted grid
    """
    if lon_min < 0:
        lon_min = 360 + lon_min
    if lon_max < 0:
        lon_max = 360 + lon_max
    if lon_min < lon_max:
        full_data, full_lats, full_lons = grid.data(
            lat1=lat_min, lat2=lat_max, lon1=lon_min, lon2=lon_max
        )
        return full_data, full_lats, full_lons
    else:
        west_data, west_lats, west_lons = grid.data(
            lat1=lat_min, lat2=lat_max, lon1=lon_min, lon2=359.5
        )
        east_data, east_lats, east_lons = grid.data(
            lat1=lat_min, lat2=lat_max, lon1=0, lon2=lon_max
        )
        full_data = np.concatenate((west_data, east_data), axis=1)
        full_lats = np.concatenate((west_lats, east_lats), axis=1)
        full_lons = np.concatenate((west_lons, east_lons), axis=1)
        return full_data, full_lats, full_lons


# specify lat domain (inclusive) and resolution (degrees)
lat_min = 40
lat_max = 50
lat_res = 0.5

# specify lon domain (inclusive) and resolution (degrees)
lon_min = 310
lon_max = 340
lon_res = 0.5

# specify time domain (inclusive) and resolution (hours)
time_min = datetime(2018, 1, 1)
time_max = datetime(2018, 12, 31, 21)
time_res = timedelta(hours=3)

# specify WAVEWATCH model grid to collect from
grid_id = "glo_30m"

# specify the name of the netCDF file to be generated
filename = "NorthAtlantic_1.nc"

# Open a netCDF file for writing
fout = Dataset("Ocean/" + filename, mode="w")
fout.set_fill_off()

# Create netCDF file dimensions
length_lat = int((lat_max + lat_res - lat_min) / lat_res)
length_lon = int((lon_max + lon_res - lon_min) / lon_res)
length_time = int((time_max + time_res - time_min) / time_res)
lat_dim = fout.createDimension("lat", length_lat)
lon_dim = fout.createDimension("lon", length_lon)
time_dim = fout.createDimension("time", length_time)

# create netCDF file variables
lat = fout.createVariable("lat", np.float32, ("lat",), compression="zlib")
lat.units = "degrees north"
lon = fout.createVariable("lon", np.float32, ("lon",), compression="zlib")
lon.units = "degrees east"
time = fout.createVariable("time", np.float64, ("time"), compression="zlib")
time.units = "hours since " + str(time_min)
hs = fout.createVariable(
    "sig_wave_height", np.float32, ("time", "lat", "lon"), compression="zlib"
)
hs.units = "meters"
tp = fout.createVariable(
    "wave_period", np.float32, ("time", "lat", "lon"), compression="zlib"
)
tp.units = "seconds"
dp = fout.createVariable(
    "wave_direction", np.float32, ("time", "lat", "lon"), compression="zlib"
)
dp.units = "degrees true"
ws = fout.createVariable(
    "wind_speed", np.float32, ("time", "lat", "lon"), compression="zlib"
)
ws.units = "meters per second"
wd = fout.createVariable(
    "wind_direction", np.float32, ("time", "lat", "lon"), compression="zlib"
)
wd.units = "degrees true"
fout.set_auto_mask(False)
fout.set_always_mask(False)

# write time variable data to netCDF file
dates = [time_min + time_res * i for i in range(length_time)]
time[:] = date2num(dates, time.units)

# extract and write data from grib files to netCDF file, fetching files
# from the NOAA server if they are not already in the current working directory
months = sorted(set([d.strftime("%Y%m") for d in dates]))
base = "https://polar.ncep.noaa.gov/waves/hindcasts/multi_1"
wroteLatLonData = False
for m in months:
    print("writing month " + m)
    dates_this_month = [d for d in dates if d.month == int(m[4:])]
    files = [
        "multi_1.{g}.{p}.{d}.grb2".format(g=grid_id, p=p_id, d=m)
        for p_id in ["hs", "tp", "dp", "wind"]
    ]
    for var, f in enumerate(files):
        if not os.path.isfile(f):
            url = "{}/{}/gribs/{}".format(base, m, f)
            r = requests.get(url)
            open(f, "wb").write(r.content)
        grib_file = pygrib.open(f)
        # write latitude and longitude data to netCDF file (just once)
        if not wroteLatLonData:
            _, sample_lats, sample_lons = extract_grid(
                lat_min, lat_max, lon_min, lon_max, grib_file.readline()
            )
            lat[:] = sample_lats[:, 0]
            lon[:] = sample_lons[0, :]
            lon[:] = [u - 360 if u > 180 else u for u in lon]
            wroteLatLonData = True
        if var == 0:  # significant wave height
            hs_grib = grib_file.select(validDate=dates_this_month)
            for message in hs_grib:
                time_index = int((message.validDate - time_min) / time_res)
                data, _, _ = extract_grid(lat_min, lat_max, lon_min, lon_max, message)
                hs[time_index, :, :] = data
        elif var == 1:  # wave period
            tp_grib = grib_file.select(validDate=dates_this_month)
            for message in tp_grib:
                time_index = int((message.validDate - time_min) / time_res)
                date, _, _ = extract_grid(lat_min, lat_max, lon_min, lon_max, message)
                tp[time_index, :, :] = data
        elif var == 2:  # wave direction
            dp_grib = grib_file.select(validDate=dates_this_month)
            for message in dp_grib:
                time_index = int((message.validDate - time_min) / time_res)
                data, _, _ = extract_grid(lat_min, lat_max, lon_min, lon_max, message)
                dp[time_index, :, :] = data
        else:  # wind - requires further calculations
            u_grib = grib_file.select(shortName="u", validDate=dates_this_month)
            v_grib = grib_file.select(shortName="v", validDate=dates_this_month)
            u_grib.sort(key=lambda x: x.validDate)
            v_grib.sort(key=lambda x: x.validDate)
            for u_message, v_message in zip(u_grib, v_grib):
                time_index = int((u_message.validDate - time_min) / time_res)
                data_u, _, _ = extract_grid(
                    lat_min, lat_max, lon_min, lon_max, u_message
                )
                data_v, _, _ = extract_grid(
                    lat_min, lat_max, lon_min, lon_max, v_message
                )
                speed, dir = convert_wind(data_u, data_v)
                ws[time_index, :, :] = speed
                wd[time_index, :, :] = dir
        grib_file.close()
fout.close()
