import sys
import os
import pytest
import numpy as np
import datetime as dt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Ocean.weather_data_interpolation_API as interpolator


# --- TESTS for get_file_metadata ---


# test 1: check if the output is a tuple in right order and format
def test_get_file_metadata_single_file_tuple_check():
    file_name = "gfsWave_Global25_20241201_00_003"
    issue_datetime = dt.datetime.strptime(f"20241201_00", "%Y%m%d_%H")
    forecast_datetime = dt.datetime.strptime(f"20241201_03", "%Y%m%d_%H")

    input_list = [file_name]
    metadata = interpolator.get_file_metadata(input_list)

    correct = (file_name, issue_datetime, forecast_datetime)

    assert len(metadata) == 1
    assert metadata[0] == correct


# test 2: check if function correctly parse all files in given file list
def test_get_file_metadata_multiple_files():
    file1 = "gfsWave_Global25_20241201_00_000"
    f1_i_dt = dt.datetime.strptime(f"20241201_00", "%Y%m%d_%H")
    f1_f_dt = dt.datetime.strptime(f"20241201_00", "%Y%m%d_%H")

    file2 = "gfsWave_Global25_20241201_00_003"
    f2_i_dt = dt.datetime.strptime(f"20241201_00", "%Y%m%d_%H")
    f2_f_dt = dt.datetime.strptime(f"20241201_03", "%Y%m%d_%H")

    file3 = "gfsWave_Global25_20241201_06_003"
    f3_i_dt = dt.datetime.strptime(f"20241201_06", "%Y%m%d_%H")
    f3_f_dt = dt.datetime.strptime(f"20241201_09", "%Y%m%d_%H")

    input_list = [file1, file2, file3]
    metadata = interpolator.get_file_metadata(input_list)

    correct = [
        (file1, f1_i_dt, f1_f_dt),
        (file2, f2_i_dt, f2_f_dt),
        (file3, f3_i_dt, f3_f_dt),
    ]

    assert len(metadata) == 3
    assert metadata == correct


# test 3: if there is a file with bad name, warning is logged
def test_get_file_metadata_malformed_filename(caplog):
    file1 = "gfsWave_20241201_00_003.nc"
    file2 = "gfsWave_Global25_20241201_00_003"
    f2_i_dt = dt.datetime.strptime(f"20241201_00", "%Y%m%d_%H")
    f2_f_dt = dt.datetime.strptime(f"20241201_03", "%Y%m%d_%H")

    input_list = [file1, file2]
    with caplog.at_level("WARNING"):
        metadata = interpolator.get_file_metadata(input_list)

    assert metadata == [(file2, f2_i_dt, f2_f_dt)]
    assert f"Skipping file {file1}" in caplog.text


# test 4: warrning is given when there is a file with bad date
def test_get_file_metadata_nonNumeric_in_filedate(caplog):
    file1 = "gfsWave_Global25_20241201_0A0_00.nc"
    file2 = "gfsWave_Global25_20241201_00_003"
    f2_i_dt = dt.datetime.strptime(f"20241201_00", "%Y%m%d_%H")
    f2_f_dt = dt.datetime.strptime(f"20241201_03", "%Y%m%d_%H")

    input_list = [file1, file2]
    with caplog.at_level("WARNING"):
        metadata = interpolator.get_file_metadata(input_list)

    assert metadata == [(file2, f2_i_dt, f2_f_dt)]
    assert f"Skipping file {file1}" in caplog.text


# --- TESTS for filter_files_by_issue_time ---


def make_entry(file_path, issue_dt, forecast_dt):
    return (file_path, issue_dt, forecast_dt)


# test 1: all issue times < current_dt → returns none
def test_filter_files_by_issue_time_earlier_issue_dt():
    current_dt = dt.datetime(2024, 12, 5, 12)
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 9)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 9), dt.datetime(2024, 12, 5, 15)
        ),
    ]
    result = interpolator.filter_files_by_issue_time(metadata, current_dt)
    assert result == []


# test 2: issue time exactly equal to current_dt → includes it
def test_filter_exact_match_included():
    current_dt = dt.datetime(2024, 12, 5, 12)
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 12), dt.datetime(2024, 12, 5, 12)
        ),
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 12), dt.datetime(2024, 12, 5, 15)
        ),
    ]
    result = interpolator.filter_files_by_issue_time(metadata, current_dt)
    assert result == metadata


# test 3: some issue times > current_dt → filters those out
def test_filter_some_entries_filtered():
    current_dt = dt.datetime(2024, 12, 5, 12)
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 15)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 12), dt.datetime(2024, 12, 5, 15)
        ),
        make_entry(
            "file3.nc", dt.datetime(2024, 12, 5, 15), dt.datetime(2024, 12, 5, 15)
        ),
    ]
    result = interpolator.filter_files_by_issue_time(metadata, current_dt)
    expected = [
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 12), dt.datetime(2024, 12, 5, 15)
        ),
    ]
    assert result == expected


# test 4: all issue times > current_dt → returns empty list
def test_filter_all_entries_filtered():
    current_dt = dt.datetime(2024, 12, 5, 12)
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 8, 6), dt.datetime(2024, 12, 8, 9)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 8, 12), dt.datetime(2024, 12, 8, 15)
        ),
    ]
    result = interpolator.filter_files_by_issue_time(metadata, current_dt)
    assert result == []


# --- TESTS for find_time_bounds ---


# test 1: fcst_dt between 2 entries → returns bounding pair
def test_find_time_bounds_found_between_entries():
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 9)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 12)
        ),
    ]
    fcst_dt = dt.datetime(2024, 12, 5, 10)
    result = interpolator.find_time_bounds(metadata, fcst_dt)
    assert result == (metadata[0], metadata[1])


# test 2: fcst_dt == lower bound
def test_find_time_bounds_exact_lower_bound():
    # result is 9:00 and 9:00
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 9)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 12)
        ),
    ]
    fcst_dt = dt.datetime(2024, 12, 5, 10)
    result = interpolator.find_time_bounds(metadata, fcst_dt)
    assert result == (metadata[0], metadata[1])


# test 3: fcst_dt == upper bound
def test_find_time_bounds_exact_upper_bound():
    # result is 12:00 and 12:00
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 9)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 12)
        ),
    ]
    fcst_dt = dt.datetime(2024, 12, 5, 10)
    result = interpolator.find_time_bounds(metadata, fcst_dt)
    assert result == (metadata[0], metadata[1])


# test 4: fcst_dt below min → raises ValueError
def test_find_time_bounds_below_min_raises():
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 9)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 12)
        ),
    ]
    fcst_dt = dt.datetime(2024, 12, 5, 6)
    with pytest.raises(ValueError, match="No bounding forecast times found"):
        interpolator.find_time_bounds(metadata, fcst_dt)


# test 5: fcst_dt above max → raises ValueError
def test_find_time_bounds_above_max_raises():
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 9)
        ),
        make_entry(
            "file2.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 12)
        ),
    ]
    fcst_dt = dt.datetime(2024, 12, 5, 15)
    with pytest.raises(ValueError, match="No bounding forecast times found"):
        interpolator.find_time_bounds(metadata, fcst_dt)


# test 6: only 1 entry → raises ValueError
def test_bounds_one_entry_raises():
    metadata = [
        make_entry(
            "file1.nc", dt.datetime(2024, 12, 5, 6), dt.datetime(2024, 12, 5, 9)
        ),
    ]
    fcst_dt = dt.datetime(2024, 12, 5, 10)
    with pytest.raises(ValueError, match="No bounding forecast times found"):
        interpolator.find_time_bounds(metadata, fcst_dt)


# test 7: empty metadata → raises ValueError
def test_bounds_empty_list_raises():
    metadata = []
    fcst_dt = dt.datetime(2024, 12, 5, 9)
    with pytest.raises(ValueError, match="No bounding forecast times found"):
        interpolator.find_time_bounds(metadata, fcst_dt)


# --- TESTS for get_surrounding_indices ---


# test 1: value between two elements — stays separate
def test_get_surrounding_indices_valid():
    array = [1, 3, 5, 7, 9]
    value = 4
    i1, i2 = interpolator.get_surrounding_indices(array, value)
    assert (i1, i2) == (1, 2)


def test_get_surrounding_indices_left_bound():
    array, value = ([1, 3, 5, 7], 1)  # value == first element
    i1, i2 = interpolator.get_surrounding_indices(array, value)
    assert (i1, i2) == (0, 1)


def test_get_surrounding_indices_right_bound():
    array, value = ([1, 3, 5, 7], 7)  # value == last element
    i1, i2 = interpolator.get_surrounding_indices(array, value)
    assert (i1, i2) == (len(array) - 2, len(array) - 1)


# out-of-bound value - raises ValueError
@pytest.mark.parametrize(
    "array, value",
    [
        ([2, 4, 6, 8], 1),  # value below first
        ([2, 4, 6, 8], 10),  # value above last
    ],
)
def test_get_surrounding_indices_indices_out_of_bounds(array, value):
    with pytest.raises(ValueError, match="out of bounds"):
        interpolator.get_surrounding_indices(array, value)


# small array with < 2 elements — raises ValueError
@pytest.mark.parametrize(
    "array, value",
    [
        ([], 5),  # empty array
        ([5], 5),  # single-element array
    ],
)
def test_get_surrounding_indices_small_array(array, value):
    with pytest.raises(ValueError, match="at least two elements"):
        interpolator.get_surrounding_indices(array, value)


# 2-element array — valid bounding and inbetween values
@pytest.mark.parametrize("value", [2, 3, 4])
def test_get_surrounding_indices_two_element_array_exact_equals_raises(value):
    array = [2, 4]
    i1, i2 = interpolator.get_surrounding_indices(array, value)
    assert (i1, i2) == (0, 1)


# --- TESTS for extract_value ---
# test 1: valid indices
def test_extract_value_valid():
    data = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ]
    )
    result = interpolator.extract_value(data, 1, 2)
    assert result == 60


# test 2: out-of-bounds indices → raises IndexError
@pytest.mark.parametrize(
    "lat_idx, lon_idx",
    [
        (3, 0),  # row out-of-bounds
        (0, 3),  # column out-of-bounds
        (5, 5),  # both out-of-bounds
    ],
)
def test_extract_value_out_of_bounds(lat_idx, lon_idx):
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(IndexError):
        interpolator.extract_value(data, lat_idx, lon_idx)


# test 3: negative index — valid
def test_extract_value_negative_index():
    data = np.array([[1, 2], [3, 4]])
    result = interpolator.extract_value(data, -1, -1)
    assert result == 4


# test 4: empty data → raises error
def test_extract_value_empty_array():
    data = np.array([[]])
    with pytest.raises(IndexError):
        interpolator.extract_value(data, 0, 0)


# test 5: masked data
def test_extract_value_masked_value_returns_nan():
    data = np.ma.array([[1, 2], [3, 4]], mask=[[0, 0], [1, 0]])  # mask [1, 0]
    val = interpolator.extract_value(data, 1, 0)
    assert np.isnan(val)


# --- TESTS for interpolate_spatial_value ---


# Fixture grid: lat = [10.0, 11.0], lon = [100.0, 101.0]
@pytest.fixture
def basic_grid():
    lats = np.array([10.0, 11.0])
    lons = np.array([100.0, 101.0])
    data = np.array([[10.0, 20.0], [30.0, 40.0]])
    return data, lats, lons


def test_interpolate_spatial_value_center_point(basic_grid):

    data, lats, lons = basic_grid
    target_lat = 10.5
    target_lon = 100.5
    corners, weights = interpolator.get_spatial_weights(
        lats, lons, target_lat, target_lon
    )
    result = interpolator.interpolate_spatial_value(data, corners, weights)
    expected = (10 + 20 + 30 + 40) / 4
    assert np.isclose(result, expected)


def test_interpolate_spatial_value_on_lat_edge(basic_grid):

    data, lats, lons = basic_grid
    target_lat = 10.0  # exact lat (edge case)
    target_lon = 100.5
    corners, weights = interpolator.get_spatial_weights(
        lats, lons, target_lat, target_lon
    )
    result = interpolator.interpolate_spatial_value(data, corners, weights)
    expected = (10 * 0.5 + 20 * 0.5) / 1
    assert np.isclose(result, expected)


def test_interpolate_spatial_value_on_lon_edge(basic_grid):
    data, lats, lons = basic_grid
    target_lat = 10.5
    target_lon = 100.0  # exact lon
    corners, weights = interpolator.get_spatial_weights(
        lats, lons, target_lat, target_lon
    )
    result = interpolator.interpolate_spatial_value(data, corners, weights)
    expected = (10 * 0.5 + 30 * 0.5) / 1
    assert np.isclose(result, expected)


def test_interpolate_spatial_value_exact_grid_point(basic_grid):

    data, lats, lons = basic_grid
    target_lat = 11.0
    target_lon = 101.0
    corners, weights = interpolator.get_spatial_weights(
        lats, lons, target_lat, target_lon
    )
    result = interpolator.interpolate_spatial_value(data, corners, weights)
    assert result == 40.0


def test_interpolate_spatial_value_with_one_masked(basic_grid):

    data, lats, lons = basic_grid
    data[0, 0] = np.nan  # mask top-left corner
    target_lat = 10.5
    target_lon = 100.5
    corners, weights = interpolator.get_spatial_weights(
        lats, lons, target_lat, target_lon
    )
    result = interpolator.interpolate_spatial_value(data, corners, weights)

    # Remaining values:
    # [0, 1] = 20
    # [1, 0] = 30
    # [1, 1] = 40
    # Each normally has weight = 0.25 → renormalized to 1/3
    # Interpolated = (20 + 30 + 40) / 3 = 30.0
    assert not np.isnan(result)
    assert np.isclose(result, 30.0)


def test_interpolate_spatial_value_with_one_masked_skewed_point(basic_grid):

    data, lats, lons = basic_grid
    data[0, 0] = np.nan  # mask top-left corner
    target_lat = 10.7
    target_lon = 100.8
    corners, weights = interpolator.get_spatial_weights(
        lats, lons, target_lat, target_lon
    )
    result = interpolator.interpolate_spatial_value(data, corners, weights)

    # lat 0 scale: 1-(10.7 - 10.0)/(11.0-10.0) = 0.3
    # lat 1 scale: 1-(11.0 - 10.7)/(11.0-10.0) = 0.7
    # lon 0 scale: 1-(100.8 - 100.0)/(101.0-100.0) = 0.2
    # lon 1 scale: 1-(101.0 - 100.8)/(101.0-100.0) = 0.8

    w_01 = 0.3 * 0.8  # [0,1] = 20
    w_10 = 0.7 * 0.2  # [1,0] = 30
    w_11 = 0.7 * 0.8  # [1,1] = 40

    weighted_sum = 20.0 * w_01 + 30.0 * w_10 + 40.0 * w_11
    total_weight = w_01 + w_10 + w_11
    expected = weighted_sum / total_weight

    assert not np.isnan(result)
    assert np.isclose(result, expected)


def test_interpolate_spatial_value_all_masked_returns_nan():

    lats = np.array([10.0, 11.0])
    lons = np.array([100.0, 101.0])
    data = np.full((2, 2), np.nan)
    target_lat = 10.5
    target_lon = 100.5
    corners, weights = interpolator.get_spatial_weights(
        lats, lons, target_lat, target_lon
    )
    result = interpolator.interpolate_spatial_value(data, corners, weights)
    assert np.isnan(result)


test_bounds_one_entry_raises()
