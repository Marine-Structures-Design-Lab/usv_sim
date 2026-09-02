from Ocean import weather_data_downsizer as downsizer
import os
import netCDF4


"""
    Run test in terminal: python -m Ocean.tests.sample_test_subset_grib
"""

INPUT_DIR = os.path.expanduser(
    "~/Documents/MSDL/core/Ocean/tests/weather_2_sample_data"
)
input_file = os.path.join(INPUT_DIR, "gfsWave_Global25_20241117_00_003.grib2")
output_file = "Ocean/tests/test_single_output.nc"

# Set a lat/lon bounding box — choose values inside your file’s coverage range!
LAT_S = 50
LAT_N = 60
LON_W = -156
LON_E = -132

# Call process_single_grib function with the input file and other parameters
downsizer.process_single_grib(input_file, output_file, LON_W, LON_E, LAT_S, LAT_N)

try:
    ds = netCDF4.Dataset("Ocean/tests/test_subset_output.nc")
    print("File is valid.")
    ds.close()
except Exception as e:
    print(f"File is invalide or correupted: {e}")
