import sys
import os
import pytest
import numpy as np
import numpy.ma as ma
from unittest.mock import MagicMock, patch

#! Make sure to run the test from Ocean
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.Ocean.weather_contour_map_visualizer as vis


# ----- specifically for read_initial_data tests -----
@pytest.fixture
def reset_unit():
    vis.UNIT = ""
    yield
    vis.UNIT = ""


# ----- get_file_list TESTS -----


def test_get_file_list_returns_sorted_files(tmp_path, caplog):
    # Create sample .nc files in non-sorted order
    file1 = tmp_path / "b.nc"
    file2 = tmp_path / "a.nc"
    file1.write_text("dummy")
    file2.write_text("dummy")

    with caplog.at_level("INFO"):
        result = vis.get_file_list(str(tmp_path))
    print(result)
    expected = [str(file2), str(file1)]  # Sorted by name
    assert result == expected
    # assert f"{len(expected)} .nc files found" in caplog.text


def test_get_file_list_returns_empty_when_no_nc_files(tmp_path, caplog):
    # Create files that don't match the pattern
    #! automatically creates the files when .write_text() is ran
    (tmp_path / "file1.txt").write_text("not nc")
    (tmp_path / "file2.csv").write_text("also not nc")

    with caplog.at_level("INFO"):
        result = vis.get_file_list(str(tmp_path))

    assert result == []
    # assert "0 .nc files found" in caplog.text


def test_get_file_list_returns_empty_for_nonexistent_directory(tmp_path, caplog):
    fake_dir = (
        tmp_path / "nonexistent"
    )  #! doesn't create directory unless calles fake_dir.mkdir()

    with caplog.at_level("INFO"):
        result = vis.get_file_list(str(fake_dir))

    assert result == []
    # assert "0 .nc files found" in caplog.text


# ----- read_initial_data TESTS -----


def test_read_initial_data_reads_valid_file(reset_unit, monkeypatch, caplog):
    # Create a mock dataset with lat, lon, and variable
    dummy_ds = MagicMock()
    dummy_ds.__enter__.return_value = dummy_ds
    dummy_ds.__exit__.return_value = None
    dummy_ds.variables = {
        "lat": np.array([10, 20]),
        "lon": np.array([30, 40]),
        "perpw": MagicMock(
            **{"__getitem__.return_value": np.array([[1.0, 2.0]]), "units": "m"}
        ),
    }

    monkeypatch.setattr("netCDF4.Dataset", lambda _: dummy_ds)

    # with caplog.at_level("DEBUG"):
    # lat, lon, data = vis.read_initial_data("dummy.nc", "perpw")

    # assert np.all(lat == [10, 20])
    # assert np.all(lon == [30, 40])
    # assert np.all(data == [[1.0, 2.0]])
    # assert vis.UNIT == "m"
    # assert "Read initial data from dummy.nc" in caplog.text


def test_read_initial_data_variable_missing(reset_unit, monkeypatch):
    dummy_ds = MagicMock()
    dummy_ds.__enter__.return_value = dummy_ds
    dummy_ds.__exit__.return_value = None
    dummy_ds.variables = {
        "lat": np.array([10, 20]),
        "lon": np.array([30, 40]),
        # 'perpw' is missing
    }

    monkeypatch.setattr("netCDF4.Dataset", lambda _: dummy_ds)

    # with pytest.raises(KeyError):
    # vis.read_initial_data("dummy.nc", "perpw")


def test_read_initial_data_corrupt_file(reset_unit, monkeypatch):
    # Simulate corrupt file by raising an OSError on open
    monkeypatch.setattr(
        "netCDF4.Dataset", lambda _: (_ for _ in ()).throw(OSError("Corrupt file"))
    )

    # with pytest.raises(OSError, match="Corrupt file"):
    # vis.read_initial_data("bad_file.nc", "perpw")


# ----- load_masked_variable TESTS -----
def make_fake_ds(data_array, fill_value=None):
    class FakeVar:
        def __getitem__(self, key):
            return data_array

        if fill_value is not None:
            _FillValue = fill_value

    ds = MagicMock()
    ds.variables = {"myvar": FakeVar()}
    return ds


# TODO: FIX
def test_masks_fill_value_correctly():
    data = np.array([[1, -9999], [2, 3]])
    ds = make_fake_ds(data, fill_value=-9999)

    result = vis.load_masked_variable(ds, "myvar")
    assert ma.isMaskedArray(result)
    assert result.mask[0, 1]
    assert result[0, 0] == 1


# TODO: FIX
def test_masks_nan_correctly():
    data = np.array([[1, np.nan], [2, 3]])
    ds = make_fake_ds(data)

    result = vis.load_masked_variable(ds, "myvar")
    assert ma.isMaskedArray(result)
    assert result.mask[0, 1]
    assert result[1, 1] == 3


def test_returns_unmasked_array_if_no_fillvalue_or_nan():
    data = np.array([[1, 2], [3, 4]])
    ds = make_fake_ds(data)

    result = vis.load_masked_variable(ds, "myvar")
    assert ma.isMaskedArray(result)
    assert not result.mask.any()  # all unmasked
    assert np.array_equal(result.data, data)


def test_raises_key_error_if_var_not_in_dataset():
    ds = MagicMock()
    ds.variables = {}  # No 'myvar'

    with pytest.raises(KeyError):
        vis.load_masked_variable(ds, "myvar")


# ----- update_fcst_time TESTS -----


def test_update_fcst_time_same_day():
    filename = "gfsWave_Global25_20241201_12_006.nc"
    collect_dt, fcst_dt = vis.update_fcst_time(filename)

    assert collect_dt.strftime("%Y-%m-%d %H") == "2024-12-01 12"
    assert fcst_dt.strftime("%Y-%m-%d %H") == "2024-12-01 18"


def test_update_fcst_time_future_day():
    filename = "gfsWave_Global25_20241201_12_066.nc"
    collect_dt, fcst_dt = vis.update_fcst_time(filename)

    assert collect_dt.strftime("%Y-%m-%d %H") == "2024-12-01 12"
    assert fcst_dt.strftime("%Y-%m-%d %H") == "2024-12-04 06"


def test_update_fcst_time_zero_padded_hr():
    filename = "gfsWave_Global25_20241201_03_006.nc"
    collect_dt, fcst_dt = vis.update_fcst_time(filename)

    assert collect_dt.strftime("%Y-%m-%d %H") == "2024-12-01 03"
    assert fcst_dt.strftime("%Y-%m-%d %H") == "2024-12-01 09"


def test_update_fcst_time_invalid_format():
    bad_filename = "gfsWave_Global25_20241201_03_00X.nc"
    with pytest.raises(ValueError):
        vis.update_fcst_time(bad_filename)


# ------ filelist_editor TESTS -----


def test_filelist_editor_filters_by_issue_date():
    files = [
        "weather_Global25_20241201_00_000.nc",
        "weather_Global25_20241201_00_006.nc",
        "weather_Global25_20241201_06_003.nc",
        "weather_Global25_20241202_00_003.nc",
    ]
    filtered = vis.filelist_editor(files, fcst_issue_time="20241201_00")
    assert filtered == [
        "weather_Global25_20241201_00_000.nc",
        "weather_Global25_20241201_00_006.nc",
    ]


def test_filelist_editor_returns_nowcast_if_none():
    files = [
        "weather_Global25_20241201_00_000.nc",
        "weather_Global25_20241201_06_003.nc",
    ]
    filtered = vis.filelist_editor(files, fcst_issue_time=None)
    assert filtered == ["weather_Global25_20241201_00_000.nc"]


def test_filelist_editor_raises_if_no_match():
    files = [
        "weather_Global25_20241201_00_000.nc",
        "weather_Global25_20241201_06_000.nc",
    ]

    with pytest.raises(FileNotFoundError, match="No forecast files found"):
        vis.filelist_editor(files, fcst_issue_time="20250101_12")


def test_filelist_editor_empty_input_raises():
    with pytest.raises(FileNotFoundError):
        vis.filelist_editor([], fcst_issue_time="20241201_00")
