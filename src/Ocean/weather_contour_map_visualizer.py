import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.Ocean.weather_utils import (
    filelist_editor,
    get_file_list,
    read_grid_info,
    read_variable_data,
    update_fcst_time,
)
import logging

# ---------- LOGGING SETUP ----------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
# TODO: update to the directory path to the folder containing .nc files
INPUT_DIR = "src/Ocean/tests/weather_2_nc/0101_0110_lat-5_5_lon270_280_fc50_ts9"

# TODO: update the string with variable interested in visualizing
VARIABLE_NAME = "mpts"
UNIT = ""  #! edited by read_initial_data function

# TODO: update the string with fcst issued at specific date and time
#! If interested in nowcasts, assign value None to variable
# FORECAST_ISSUE_TIME = None  # e.g. "20241201_12"
FORECAST_ISSUE_TIME = "20250101_00"


def load_masked_variable(ds, varname):
    """
    Requires: ds is an open NetCDF4 dataset. varname is a variable in ds.

    Modifies: None.

    Effects: Returns a masked array of varname, where _FillValue or NaNs are masked out.
    """

    var = ds.variables[varname]
    data = var[:]
    fill_value = getattr(var, "_FillValue", np.nan)
    masked_data = np.ma.masked_where((data == fill_value) | np.isnan(data), data)
    return masked_data


def update_frame(frame, file_list, mesh, title, VARIABLE_NAME):
    """
    Requires: frame is a valid index into file_list. Files in file_list are NetCDF files with the variable VARIABLE_NAME.

    Modifies: mesh, title.

    Effects: Updates mesh and title to reflect the data and forecast time for the current frame.
                Returns updated mesh and title. Logs debug information about the current frame.
    """

    file_path = file_list[frame]
    with nc.Dataset(file_path) as ds:
        data = load_masked_variable(ds, VARIABLE_NAME)
        mesh.set_array(data.ravel())

    file_name = os.path.basename(file_path)
    issue_datetime, fcst_datetime = update_fcst_time(file_name)

    logger.debug(
        f"Frame {frame}: Forecast time = {fcst_datetime}, Issued = {issue_datetime}"
    )

    if issue_datetime == fcst_datetime:
        title.set_text(
            f"{VARIABLE_NAME.upper()} on {fcst_datetime.strftime('%Y%m%d at %H:%M')}"
        )
    else:
        title.set_text(
            f"{VARIABLE_NAME.upper()} on {fcst_datetime.strftime('%Y%m%d at %H:%M')} "
            f"issued on {issue_datetime.strftime('%Y%m%d at %H:%M')}"
        )

    return mesh, title


# ---------- ANIMATION GENERATOR ----------
def create_animation(file_list, VARIABLE_NAME, output_path, fcst_issue_time, fps=2):
    """
    Requires: file_list is a list of valid NetCDF file paths.
                VARIABLE_NAME is a valid variable in those files.
                output_path is a writable video file path.
                fcst_issue_time
                fps is a positive integer (frames per second) defaulted to 2.

    Modifies: File at output_path.

    Effects: Filters file list, sets up the figure and animation, and saves the resulting animation as a video file.
                Logs progress and success messages. Displays the animation.
    """

    logger.info(f"Creating animation: {output_path}")
    file_list = filelist_editor(file_list, fcst_issue_time)
    lat_array, lon_array, _, _, _, _, UNIT = read_grid_info(file_list[0], VARIABLE_NAME)
    data = read_variable_data(file_list[0], VARIABLE_NAME)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="darkgray")

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(lon_array, lat_array, data, shading="auto", cmap=cmap)
    plt.colorbar(mesh, ax=ax, label=f"{VARIABLE_NAME.upper()} ({UNIT})")
    title = ax.set_title("")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ani = animation.FuncAnimation(
        fig,
        update_frame,
        fargs=(file_list, mesh, title, VARIABLE_NAME),
        frames=len(file_list),
        blit=False,
        repeat=False,
    )

    ani.save(output_path, writer="ffmpeg", fps=fps)
    logger.info("✅ Animation saved successfully.")
    plt.show()


# ---------- MAIN ----------
def main():
    logger.info(f"Working directory: {os.getcwd()}")
    output_video = f"{INPUT_DIR}{VARIABLE_NAME}_fcst{FORECAST_ISSUE_TIME}_animation.mp4"

    file_list = get_file_list(INPUT_DIR)

    if not file_list:
        logger.error(f"No .nc files found in {INPUT_DIR}")
        raise FileNotFoundError(f"No .nc files found in: {INPUT_DIR}")

    create_animation(file_list, VARIABLE_NAME, output_video, FORECAST_ISSUE_TIME)


if __name__ == "__main__":
    main()
