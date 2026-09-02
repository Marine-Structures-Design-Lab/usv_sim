from Ocean.template_interface import OceanMod
from datetime import datetime, timedelta
from random import random, choice, sample
from netCDF4 import Dataset
import numpy as np
import pytest
import os


def create_sample_NOAA_grid():
    fout = Dataset("Ocean/NOAA_test_suite_grid.nc", mode="w")
    fout.set_fill_off()
    lat_dim = fout.createDimension("lat", 10)
    lon_dim = fout.createDimension("lon", 10)
    time_dim = fout.createDimension("time", 10)
    lat = fout.createVariable("lat", np.float32, ("lat",), compression="zlib")
    lat.units = "degrees north"
    lon = fout.createVariable("lon", np.float32, ("lon",), compression="zlib")
    lon.units = "degrees east"
    time = fout.createVariable("time", np.float32, ("time",), compression="zlib")
    time.units = "hours since 2018-01-01 00:00:00"
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

    lat[:] = range(0, 10)
    lon[:] = range(0, 10)
    time[:] = range(0, 10)

    for i in range(10):
        for j in range(10):
            for k in range(10):
                hs[i, j, k] = i + 2 * j + 3 * k
                tp[i, j, k] = 2 * i + 3 * j + k
                dp[i, j, k] = i + j + k
                ws[i, j, k] = 3 * i + j + 2 * k
                wd[i, j, k] = 3 * i + 2 * j + k

    fout.close()


def create_sample_n_var_grid():
    fout = Dataset("Ocean/sample_n_var_grid.nc", mode="w")
    fout.set_fill_off()
    lat_dim = fout.createDimension("lat", 5)
    lon_dim = fout.createDimension("lon", 5)
    time_dim = fout.createDimension("time", 5)
    lat = fout.createVariable("lat", np.float32, ("lat",), compression="zlib")
    lat.units = "degrees north"
    lon = fout.createVariable("lon", np.float32, ("lon",), compression="zlib")
    lon.units = "degrees east"
    time = fout.createVariable("time", np.float32, ("time",), compression="zlib")
    time.units = "hours since 2018-01-01 00:00:00"

    lat[:] = range(0, 5)
    lon[:] = range(0, 5)
    time[:] = range(0, 5)

    vars = []
    n = choice(range(1, 7))
    names = sample(["a", "b", "c", "d", "e", "f"], n)
    data = np.zeros((5, 5, 5))
    for i in range(n):
        vars.append(
            fout.createVariable(
                names[i], np.float32, ("time", "lat", "lon"), compression="zlib"
            )
        )
        vars[i].units = "fill_value"
        vars[i][:] = data
    fout.close()


def test_mission_before_loading():
    model = OceanMod()
    with pytest.raises(AttributeError):
        model.weather_mission([datetime.now()], [(0, 0)])


def test_mission_diff_length_inputs():
    model = OceanMod()
    if not os.path.isfile("Ocean/NOAA_test_suite_grid.nc"):
        create_sample_NOAA_grid()
    model.load_data("Ocean/NOAA_test_suite_grid.nc")
    with pytest.raises(ValueError):
        model.weather_mission([datetime(2018, 1, 1, hour=2)], [(0, 0), (1, 1)])


def test_mission_invalid_inputs():
    model = OceanMod()
    if not os.path.isfile("Ocean/NOAA_test_suite_grid.nc"):
        create_sample_NOAA_grid()
    model.load_data("Ocean/NOAA_test_suite_grid.nc")
    with pytest.raises(ValueError):
        model.weather_mission([datetime(2017, 12, 31)], [(1, 2)])
    with pytest.raises(ValueError):
        model.weather_mission([datetime(2018, 2, 1)], [(1, 2)])
    with pytest.raises(ValueError):
        model.weather_mission([datetime(2018, 1, 1)], [(-1, 1)])
    with pytest.raises(ValueError):
        model.weather_mission([datetime(2018, 1, 1)], [(11, 1)])
    with pytest.raises(ValueError):
        model.weather_mission([datetime(2018, 1, 1)], [(1, -1)])
    with pytest.raises(ValueError):
        model.weather_mission([datetime(2018, 1, 1)], [(1, 11)])


def test_load_data():
    model = OceanMod()
    params = model.get_params()
    units = model.get_units()
    if not os.path.isfile("Ocean/NOAA_test_suite_grid.nc"):
        create_sample_NOAA_grid()
    model.load_data("Ocean/NOAA_test_suite_grid.nc")
    assert params["lat"] == (0.0, 9.0)
    assert units["lat"] == "degrees north"


def interpolate_NOAA(date, coordinate):
    date = (date - datetime(2018, 1, 1)) / timedelta(hours=1)
    lat = coordinate[0]
    lon = coordinate[1]
    hs = date + 2 * lat + 3 * lon
    tp = 2 * date + 3 * lat + lon
    dp = date + lat + lon
    ws = 3 * date + lat + 2 * lon
    wd = 3 * date + 2 * lat + lon
    return (hs, dp, tp, wd, ws)


def test_NOAA_interpolation():
    model = OceanMod()
    if not os.path.isfile("Ocean/NOAA_test_suite_grid.nc"):
        create_sample_NOAA_grid()
    model.load_data("Ocean/NOAA_test_suite_grid.nc")
    dates = [datetime(2018, 1, 1) + timedelta(hours=(random() * 9)) for _ in range(10)]
    coordinates = [(random() * 9, random() * 9) for _ in range(10)]
    outputs, names = model.weather_mission(dates, coordinates)
    for d, c, output in zip(dates, coordinates, outputs):
        my_output = interpolate_NOAA(d, c)
        assert abs(my_output[0] - output[0]) <= 0.1
        assert abs(my_output[1] - output[1]) <= 0.1
        assert abs(my_output[2] - output[2]) <= 0.1
        assert abs(my_output[3] - output[3]) <= 0.1
        assert abs(my_output[4] - output[4]) <= 0.1
    assert names == [
        "sig_wave_height",
        "wave_direction",
        "wave_period",
        "wind_direction",
        "wind_speed",
    ]


def test_n_variable_file():
    model = OceanMod()
    create_sample_n_var_grid()
    model.load_data("Ocean/sample_n_var_grid.nc")
    fin = Dataset("Ocean/sample_n_var_grid.nc")
    param_names = sorted(list(model.get_params().keys()))
    assert param_names == ["lat", "lon", "time"]
    unit_names = sorted(list(model.get_units().keys()))
    assert unit_names == sorted(list(fin.variables.keys()))
    _, vars = model.weather_mission([datetime(2018, 1, 1, 3)], [(1, 1)])
    var_names = [u for u in unit_names if u != "time" and u != "lat" and u != "lon"]
    assert vars == var_names
    os.remove("Ocean/sample_n_var_grid.nc")


def test_invalid_plot_grid():
    model = OceanMod()
    if not os.path.isfile("Ocean/NOAA_test_suite_grid.nc"):
        create_sample_NOAA_grid()
    model.load_data("Ocean/NOAA_test_suite_grid.nc")
    with pytest.raises(ValueError):
        model.plot_grid(datetime(2018, 1, 1), "air")
    with pytest.raises(ValueError):
        model.plot_grid(datetime(2018, 1, 1), "pressure")
    with pytest.raises(ValueError):
        model.plot_grid(datetime(2019, 3, 4), "wind")
