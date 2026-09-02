### Takes EU grib files and converts them into NetCDF file
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview


import cdsapi
import pygrib
import netCDF4
import numpy as np
import datetime as dt
import requests
import math
import os
from urllib.request import urlopen


def convert_wind(wind_u_comp, wind_v_comp):
    """
    Convert wind from component form to magnitude and direction form.


    Parameters
    ----------
    :param wind_u_comp: float
    west to east component of wind speed
    :param wind_v_comp: float
    south to north component of wind speed


    Returns
    -------
    wind_speed : float
    magnitude of wind speed
    wind_dir : float
    direction of wind in degrees true
    """
    # # calculate speed using Euclidean distance formula
    # wind_speed = math.sqrt(wind_u_comp**2 + wind_v_comp**2)
    # # calculate destination direction in degrees relative
    # # to the positive x-axis in a Cartesian plane (counterclockwise)
    # math_deg_towards = math.degrees(math.atan2(wind_v_comp, wind_u_comp))
    # # convert from destination direction to source direction
    # math_deg_from = (math_deg_towards + 180) % 360
    # # convert from counterclockwise measure relative to the
    # # positive x-axis to clockwise measure relative to true north
    # wind_dir = (450 - math_deg_from) % 360
    # return wind_speed, wind_dir

    wind_speed = math.sqrt(wind_u_comp**2 + wind_v_comp**2)

    # Calculate wind direction
    wind_direction = math.atan2(wind_v_comp, wind_u_comp) * (180 / math.pi)
    # Convert wind direction to a value between 0 and 360 degrees
    wind_direction = (wind_direction + 360) % 360
    return wind_speed, wind_direction


def extract_grid(lat_min, lat_max, lon_min, lon_max, grid):
    """
    Given a min/max value for lat and lon and a grib message, return
    the data grid, latitudes grid, and longitudes grid


    Parameters
    ----------
    :param lat_min: float
    minimum latitude value for grid
    :param lat_max: float
    maximum latitude value for grid
    :param lon_min: float
    minimum longitude value for grid
    :param lon_max: float
    maximum longitude value for grid
    :param grid: pygrib._pygrib.gribmessage
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
    # if lon_min < 0:
    #   lon_min = 360 + lon_min
    # if lon_max < 0:
    #   lon_max = 360 + lon_max
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


c = cdsapi.Client()


# specify lat/lon domain (inclusive)


# north atlantic
# lat_min = 40
# lat_max = 50
# lat_res = 0.5
# lon_min = -50
# lon_max = -20
# lon_res = 0.5


# north pacific
lat_min = 20
lat_max = 50
lat_res = 0.5
lon_min = -175
lon_max = -150
lon_res = 0.5


# # south pacific
# lat_min = -30
# lat_max = 0
# lat_res = 0.5
# lon_min = -175
# lon_max = -154
# lon_res = 0.5


# specify time range (inclusive) and frequency of collection
start = dt.datetime(2018, 1, 1)
end = dt.datetime(2018, 12, 31, 21)


delta = dt.timedelta(hours=3)


num_time = int(((end + delta) - start) / delta)
dates = [start + delta * d for d in range(num_time)]
months = sorted(set([d.strftime("%Y%m") for d in dates]))


# fout = netCDF4.Dataset('north_atlantic.nc', mode='w')
fout = netCDF4.Dataset("north_pacific_compress.nc", mode="w")
# fout = netCDF4.Dataset('south_pacific.nc', mode='w')


fout.set_fill_off()


length_lat = int((lat_max + lat_res - lat_min) / lat_res)
length_lon = int((lon_max + lon_res - lon_min) / lon_res)
length_time = int((end + delta - start) / delta)


print("creating dimensions...")
lat_dim = fout.createDimension("lat", length_lat)
lon_dim = fout.createDimension("lon", length_lon)
time_dim = fout.createDimension("time", length_time)


print("creating variables...")
lat = fout.createVariable("lat", np.float32, ("lat",), compression="zlib")
lat.units = "degrees north"
lon = fout.createVariable("lon", np.float32, ("lon",), compression="zlib")
lon.units = "degrees east"
time = fout.createVariable("time", np.float64, ("time"), compression="zlib")
time.units = "hours since " + str(start)
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
wd = fout.createVariable(
    "wind_direction", np.float32, ("time", "lat", "lon"), compression="zlib"
)
ws.units = "meters per second"
wd.units = "degrees true"


ap = fout.createVariable(
    "surface_pressure", np.float32, ("time", "lat", "lon"), compression="zlib"
)
ap.units = "pascal"


fout.set_auto_mask(False)
fout.set_always_mask(False)


dates = [start + delta * i for i in range(length_time)]
time[:] = netCDF4.date2num(dates, time.units)


# grib file for specific region
# ds = pygrib.open('north_atlantic.grib')
ds = pygrib.open("north_pacific.grib")
# ds = pygrib.open('south_pacific.grib')


_, sample_lats, sample_lons = extract_grid(
    lat_min, lat_max, lon_min, lon_max, ds.readline()
)


lat[:] = sample_lats[::2, 0]
lon[:] = sample_lons[0, ::2]


# lat[:] = sample_lats[:, 0]
# lon[:] = sample_lons[0, :]


try:
    print("writing significant wave height data...")
    fin_hs = sorted(
        ds.select(
            name="Significant height of combined wind waves and swell", validDate=dates
        ),
        key=lambda x: x.validDate,
    )

    for grid in fin_hs:
        time_index = int((grid.validDate - start) / delta)
        print(time_index)

        data, _, why = extract_grid(lat_min, lat_max, lon_min, lon_max, grid)
        hs[time_index, :, :] = data
except ValueError:
    print("No singificant wave height data found")


try:
    print("writing wave period data...")
    fin_tp = sorted(
        ds.select(name="Peak wave period", validDate=dates), key=lambda x: x.validDate
    )
    for grid in fin_tp:
        time_index = int((grid.validDate - start) / delta)
        data, _, _ = extract_grid(lat_min, lat_max, lon_min, lon_max, grid)
        tp[time_index, :, :] = data
except ValueError:
    print("No wave period data found")


try:
    print("writing wave direction data...")
    fin_dp = sorted(
        ds.select(name="Mean wave direction", validDate=dates),
        key=lambda x: x.validDate,
    )
    for grid in fin_dp:
        time_index = int((grid.validDate - start) / delta)
        data, _, _ = extract_grid(lat_min, lat_max, lon_min, lon_max, grid)
        dp[time_index, :, :] = data.data
except ValueError:
    print("No wave direction data found")


try:
    print("writing atmospheric pressure data...")
    fin_ap = sorted(
        ds.select(name="Surface pressure", validDate=dates), key=lambda x: x.validDate
    )
    for grid in fin_ap:
        time_index = int((grid.validDate - start) / delta)
        data, _, _ = extract_grid(lat_min, lat_max, lon_min, lon_max, grid)
        ap_data = data.data
        ind_out = 0
        ind_in = 0

        for j in range(0, ap_data.shape[0], 2):
            for k in range(0, ap_data.shape[1], 2):
                ap[time_index, ind_out, ind_in] = ap_data[j, k]
                ind_in += 1
            ind_in = 0
            ind_out += 1
except ValueError:
    print("No atmospheric pressure data found")


try:
    print("writing wave period and direction data...")
    fin_u = sorted(
        ds.select(name="10 metre U wind component", validDate=dates),
        key=lambda x: x.validDate,
    )
    fin_v = sorted(
        ds.select(name="10 metre V wind component", validDate=dates),
        key=lambda x: x.validDate,
    )
    for u_grid, v_grid in zip(fin_u, fin_v, strict=True):
        time_index = int((u_grid.validDate - start) / delta)
        data_u, lat_u, lon_u = extract_grid(lat_min, lat_max, lon_min, lon_max, u_grid)
        data_v, lat_v, lon_v = extract_grid(lat_min, lat_max, lon_min, lon_max, v_grid)
        u_grid = data_u.data
        v_grid = data_v.data

        ind_out = 0
        ind_in = 0
        for j in range(0, (u_grid.shape[0]), 2):
            for k in range(0, u_grid.shape[1], 2):
                speed, dir = convert_wind(u_grid[j, k], v_grid[j, k])
                # print(i, ind_out, ind_in)
                ws[time_index, ind_out, ind_in] = speed
                wd[time_index, ind_out, ind_in] = dir
                ind_in += 1
            ind_in = 0
            ind_out += 1
except ValueError:
    print("No wave component data found")


print("done")
ds.close()


# more generic simulations for this, use this for
# forcst local weather 24/48 hours, boat is on its own
