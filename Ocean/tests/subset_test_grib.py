import sys
import os
import netCDF4

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Ocean import weather_data_downsizer as downsizer


"""
    Run test in terminal: python -m Ocean.tests.sample_test_subset_grib
"""

INPUT_DIR = os.path.expanduser("/Volumes/Weather_2/NOAA_GFS")
input_file = os.path.join(INPUT_DIR, "*.grib2")
OUTPUT_DIR = os.path.expanduser("~/Documents/MSDL/core/Ocean/tests/weather_2_nc")

# Set a lat/lon bounding box — choose values inside your file’s coverage range!


# # With Empty LAND Values
# START_DATE = "20241201"
# END_DATE = "20241210"
# LAT_S = 50
# LAT_N = 60
# LON_W = -140
# LON_E = -130
# FORECAST_HRZ = 50
# TIMESTEP = 9

# With Empty LAND Values
START_DATE = "20241201"
END_DATE = "20241210"
LAT_S = 50
LAT_N = 60
LON_W = -140
LON_E = -130
FORECAST_HRZ = 50
TIMESTEP = 9


# Extract MMDD from dates
start_mmdd = START_DATE[4:]
end_mmdd = END_DATE[4:]

# Calculate lon/lat ranges
lat_range = abs(LAT_N - LAT_S)
lon_range = abs(LON_E - LON_W)

# Construct directory name
dir_suffix = f"{start_mmdd}_{end_mmdd}_lat{lat_range}_lon{lon_range}_fc{FORECAST_HRZ}_ts{TIMESTEP}"
base_dir = os.path.expanduser("~/Documents/MSDL/core/Ocean/tests/weather_2_nc")
OUTPUT_DIR = os.path.join(base_dir, dir_suffix)

# Call process_single_grib function with the input file and other parameters
downsizer.process_grib_directory(START_DATE, END_DATE, LAT_S, LAT_N, LON_W, LON_E, FORECAST_HRZ, TIMESTEP, INPUT_DIR, OUTPUT_DIR)