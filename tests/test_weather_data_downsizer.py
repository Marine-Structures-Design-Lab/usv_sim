import sys
import os
import pytest
import datetime as dt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Ocean.weather_data_downsizer as downsizer


# lon_lat_range_check tests
def test_lat_lon_range_check_valid_0360_convention():
    downsizer.lat_lon_range_check(-45, 45, 10, 350)


def test_lat_lon_range_check_valid_minus180_180_convention():
    downsizer.lat_lon_range_check(-45, 45, -170, 170)


def test_lat_lon_range_check_both_lon_in_valid_shared_range():
    # Should work if both are between 0 and 180
    downsizer.lat_lon_range_check(-45, 45, 50, 100)


def test_lat_lon_range_check_mixed_convention_high_low():
    with pytest.raises(ValueError, match="convention mismatch"):
        downsizer.lat_lon_range_check(-45, 45, -100, 250)


def test_lat_lon_range_check_mixed_convention_low_high():
    with pytest.raises(ValueError, match="convention mismatch"):
        downsizer.lat_lon_range_check(-45, 45, 250, -100)


def test_lat_lon_range_check_lon_w_greater_than_lon_e():
    with pytest.raises(ValueError, match="lon_w.*must be less than lon_e"):
        downsizer.lat_lon_range_check(-45, 45, 240, 200)


def test_lat_lon_range_check_lat_s_greater_than_lat_n():
    with pytest.raises(ValueError, match="lat_s.*must be less than lat_n"):
        downsizer.lat_lon_range_check(50, 40, 10, 100)


def test_lat_lon_range_check_lat_s_out_of_range():
    with pytest.raises(ValueError, match="lat_s.*between -90 and 90"):
        downsizer.lat_lon_range_check(-100, 40, 10, 100)


def test_lat_lon_range_check_lat_n_out_of_range():
    with pytest.raises(ValueError, match="lat_n.*between -90 and 90"):
        downsizer.lat_lon_range_check(-40, 100, 10, 100)


def test_lat_lon_range_check_lon_w_equal_lon_e():
    with pytest.raises(ValueError, match="lon_w.*must be less than lon_e"):
        downsizer.lat_lon_range_check(-45, 45, 50, 50)


def test_lat_lon_range_check_lat_s_equal_lat_n():
    with pytest.raises(ValueError, match="lat_s.*must be less than lat_n"):
        downsizer.lat_lon_range_check(30, 30, 10, 100)


# show_region_on_map tests


# extract_date_from_filename tests
def test_extract_date_from_filename_valid():
    assert downsizer.extract_date_from_filename(
        "gfsWave_Global25_20241117_00_024.grib2"
    ) == dt.datetime(2024, 11, 17)


def test_extract_date_from_filename_invalid_format():
    assert (
        downsizer.extract_date_from_filename("gfsWave_Global25_2024-11-17_00_024.grib2")
        is None
    )


def test_extract_date_from_filename_no_date():
    assert (
        downsizer.extract_date_from_filename("gfsWave_Global25_nodate_00_024.grib2")
        is None
    )


# extract_forecast_hour_from_filename tests
def test_extract_forecast_hour_from_filename_valid():
    assert (
        downsizer.extract_forecast_hour_from_filename(
            "gfsWave_Global25_20241117_00_024.grib2"
        )
        == 24
    )


def test_extract_forecast_hour_from_filename_invalid():
    assert (
        downsizer.extract_forecast_hour_from_filename(
            "gfsWave_Global25_20241117_00_ABC.grib2"
        )
        is None
    )


def test_extract_forecast_hour_from_filename_middle_invalid():
    assert (
        downsizer.extract_forecast_hour_from_filename(
            "gfsWave_Global25_20241117_002_ABC.grib2"
        )
        is None
    )


def test_extract_forecast_hour_from_filename_missing():
    assert (
        downsizer.extract_forecast_hour_from_filename(
            "gfsWave_Global25_2024-11-17_00.grib2"
        )
        is None
    )


# subset_grib tests
# See "sample_test_subset_grib.py" file
