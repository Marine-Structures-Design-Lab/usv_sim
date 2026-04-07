
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Ocean.weather_utils import filelist_editor, get_file_list, read_grid_info, extract_value, extract_timestamp
import logging

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
INPUT_DIR = 'Ocean/tests/weather_2_nc/1210_1231_lat30_40_lon150_160_fc0_ts6'
VARIABLE_NAME = 'swh'
UNIT = '' # UPDATED in code
TARGET_LAT =35
TARGET_LON = 160

FORECAST_PREVIEW_TIME = None # e.g., '20241210_12' for forecast; None for nowcast

# ---------- MAIN WORKFLOW ----------
def main():
    """
    Requires: Global config settings (paths, variable name, coordinates)
    Modifies: Global log output, UNIT
    Effects:
        - Reads forecast/nowcast files
        - Extracts a variable over time at a grid point
        - Creates and displays a time series line plot
    """
    file_list = get_file_list(INPUT_DIR)
    file_list = filelist_editor(file_list, FORECAST_PREVIEW_TIME)
    read_grid_info(file_list[0], VARIABLE_NAME)

    # ✅ Check if location is within grid coverage using the first file
    try:
        _, grid_lat, grid_lon = extract_value(file_list[0], VARIABLE_NAME, TARGET_LAT, TARGET_LON)
    except ValueError as e:
        logger.error(f"❌ Cannot plot: {e}")
        return

    logger.info(f"TASK: Create time series plot for variable '{VARIABLE_NAME}' at ({TARGET_LAT:.2f}°, {TARGET_LON:.2f}°)")

    times = []
    values = []

    for file_path in file_list:
        timestamp = extract_timestamp(file_path)
        if timestamp is None:
            continue
        try:
            value, _, _ = extract_value(file_path, VARIABLE_NAME, TARGET_LAT, TARGET_LON)
            values.append(value)
            times.append(timestamp)
        except Exception as e:
            logger.warning(f"⚠️ Error in {file_path}: {e}")

    if not times:
        logger.error("❌ No valid data to plot.")
        return

    # ---------- PLOTTING ----------
    mode_label = "Nowcast" if FORECAST_PREVIEW_TIME is None else f"Forecast {FORECAST_PREVIEW_TIME}"
    plt.figure(figsize=(10, 5))
    plt.plot(times, values, marker='o')
    plt.title(f'{VARIABLE_NAME.upper()} ({mode_label}) at ({grid_lat:.2f}°, {grid_lon:.2f}°)')
    plt.xlabel('Time')
    plt.ylabel(f"{VARIABLE_NAME} ({UNIT})")
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    logger.info("✅ Plotting complete.")

if __name__ == '__main__':
    main()
