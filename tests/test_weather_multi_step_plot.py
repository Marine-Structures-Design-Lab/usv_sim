import sys
import os
import pytest
import numpy as np
import numpy.ma as ma
from unittest.mock import MagicMock, patch

#! Make sure to run the test from Ocean
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Ocean.weather_multi_step_plot as msplt

# ------ find_nearest_index TESTS -----
def test_find_nearest_index_value_doesnot_exist():
    #i dont think this function exists anymore
    array = [1, 3, 6, 7]
    #output_idx = msplt.find_nearest_index(array, 4)

    #assert output_idx == 1

def test_find_nearest_index_value_exists_out_of_order():
    #i dont think this function exists anymore
    array = [1, 3, 6, 4]
    #output_idx = msplt.find_nearest_index(array, 4)

    #assert output_idx == 3

def test_find_nearest_index_value_exists_very_far():
    #i don't think this function exists anymore
    array = [1, 3, 6, 4]
    #output_idx = msplt.find_nearest_index(array, 100)

    #assert output_idx == 2

# ------ extract_timestamp TESTS -----

def test_extract_timestamp_valid_filename():
    filename = "gfsWave_Global25_20241201_12_006.nc"
    extracted_dt = msplt.extract_timestamp(filename)
    assert extracted_dt.strftime("%Y%m%d-%H") == "20241201-18"

def test_extract_timestamp_invalid_filename(caplog):
    invalid_filename = "invalid.nc"
    
    with caplog.at_level("WARNING"):
        result = msplt.extract_timestamp(invalid_filename)

    assert result == None
    assert "Skipped: could not parse time" in caplog.text

# ------ extract_value TESTS -----
@patch("netCDF4.Dataset")
def test_extract_value_lat_out_of_bound(mock_dataset_class):
    
    mock_ds = MagicMock()
    mock_dataset_class.return_value.__enter__.return_value = mock_ds
    
    # Mock lat/lon arrays
    mock_ds.variables = {
        'lat': np.array([10.0, 20.0, 30.0]),
        'lon': np.array([100.0, 110.0, 120.0]),
        'wave': MagicMock()
    }

    mock_ds.variables['wave'].__getitem__.return_value = np.array([
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
        [3.1, 3.2, 3.3]
    ])
    mock_ds.variables['wave'].ndim = 2 #2d, using lat and lon to access

    # Define global attributes (e.g., bounding box)
    mock_ds.getncattr.side_effect = lambda key: {
        'lat_min': 10.0,
        'lat_max': 30.0,
        'lon_min': 100.0,
        'lon_max': 120.0
    }[key]

    # Now use lat/lon that are outside the bounds
    #this function doesn't return that error anymore
    #with pytest.raises(ValueError, match="outside dataset bounds"):
        #msplt.extract_value("dummy_path.nc", "wave", 50.0, 120.0)

@patch("netCDF4.Dataset")
def test_extract_value_lon_out_of_bound(mock_dataset_class):
    mock_ds = MagicMock()
    mock_dataset_class.return_value.__enter__.return_value = mock_ds
    
    # Mock lat/lon arrays
    mock_ds.variables = {
        'lat': np.array([10.0, 20.0, 30.0]),
        'lon': np.array([100.0, 110.0, 120.0]),
        'wave': MagicMock()
    }

    mock_ds.variables['wave'].__getitem__.return_value = np.array([
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
        [3.1, 3.2, 3.3]
    ])
    mock_ds.variables['wave'].ndim = 2 #2d, using lat and lon to access

    # Define global attributes (e.g., bounding box)
    mock_ds.getncattr.side_effect = lambda key: {
        'lat_min': 10.0,
        'lat_max': 30.0,
        'lon_min': 100.0,
        'lon_max': 120.0
    }[key]

    # Now use lat/lon that are outside the bounds
     #this function doesn't return that error anymore
    #with pytest.raises(ValueError, match="outside dataset bounds"):
        #msplt.extract_value("dummy_path.nc", "wave", 20.0, 150.0)

@patch("netCDF4.Dataset")
def test_extract_value_lat_lon_on_border_success(mock_dataset_class):
    global UNIT
    
    mock_ds = MagicMock()
    mock_dataset_class.return_value.__enter__.return_value = mock_ds
    
    # Mock lat/lon arrays
    mock_ds.variables = {
        'lat': np.array([10.0, 20.0, 30.0]),
        'lon': np.array([100.0, 110.0, 120.0]),
        'wave': MagicMock()
    }

    mock_ds.variables['wave'].__getitem__.return_value = np.array([
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
        [3.1, 3.2, 3.3]
    ])
    mock_ds.variables['wave'].ndim = 2 #2d, using lat and lon to access
    mock_ds.variables['wave'].units = 'm'

    # Define global attributes (e.g., bounding box)
    mock_ds.getncattr.side_effect = lambda key: {
        'lat_min': 10.0,
        'lat_max': 30.0,
        'lon_min': 100.0,
        'lon_max': 120.0
    }[key]

    # Use a target near 21.0°N and 111.0°E → should round to [1, 1] = 2.2
    val, lat_used, lon_used = msplt.extract_value("dummy_path.nc", "wave", 21.0, 111.0)

    # Assertions
    assert val == 2.2
    assert lat_used == 20.0
    assert lon_used == 110.0
    #TODO: update the unit
    #assert msplt.UNIT == "m"
# ------ filelist_editor TESTS -----

def test_filelist_editor_filters_by_issue_dt():
    files = [
        "weather_Global25_20241201_00_000.nc",
        "weather_Global25_20241201_00_006.nc",
        "weather_Global25_20241201_06_003.nc",
        "weather_Global25_20241202_00_003.nc"
    ]
    filtered = msplt.filelist_editor(files, fcst_issue_time="20241201_00")
    assert filtered == ["weather_Global25_20241201_00_000.nc",
                        "weather_Global25_20241201_00_006.nc"
                        ]

def test_filelist_editor_returns_nowcast_if_none():
    files = [
        "weather_Global25_20241201_00_000.nc",
        "weather_Global25_20241201_06_003.nc",
    ]
    filtered = msplt.filelist_editor(files, fcst_issue_time=None)
    assert filtered == [ "weather_Global25_20241201_00_000.nc"
                        ]

def test_filelist_editor_raises_if_no_match():
    files = [
        "weather_Global25_20241201_00_000.nc",
        "weather_Global25_20241201_06_000.nc"
    ]
    with pytest.raises(FileNotFoundError, match="No forecast files found"):
        msplt.filelist_editor(files, fcst_issue_time="20250101_12")

def test_filelist_editor_empty_input_raises():
    with pytest.raises(FileNotFoundError):
        msplt.filelist_editor([], fcst_issue_time="20241201_00")
test_extract_value_lat_lon_on_border_success()