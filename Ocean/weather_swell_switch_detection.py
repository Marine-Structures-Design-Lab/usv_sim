import xarray as xr
import numpy as np
import os

# need to install dask


# Only get swell wave data

# Detection? - visual, or by number?
    # - find 
    # - direction
    # - same location through time
    # - hinecast, detect swell based on 
    # - utilize function, plot out the scores
    # - 


FILE_PATTERN = "Ocean/tests/weather_2_nc/0101_0110_lat-5_5_lon270_280_fc0_ts6/*_000.nc"

def angle_diff(a, b):
    """Compute smallest angular difference (degrees) between two arrays a and b."""
    diff = np.abs(a - b) % 360
    return np.minimum(diff, 360 - diff)

def main():
    ds = xr.open_mfdataset(FILE_PATTERN, combine="by_coords")
    print(ds)

    # Use the swell direction variable to work with 
    swdir = ds["swdir"]

    level = 1  # pick level
    threshold = 45.0

    lat_size = swdir.lat.size
    lon_size = swdir.lon.size
    time_size = swdir.time.size
    level = swdir.level.size

    # Arrays to store results per grid point
    switches_count = np.zeros((lat_size, lon_size), dtype=int)
    total_count = time_size - 1
    for level_idx in range(level):
        for lat_idx in range(lat_size):
            for lon_idx in range(lon_size):
                angles = swdir[:, level_idx, lat_idx, lon_idx].values
                # Count number of > threshold differences

                # DEBUG
                # for j in range(len(angles)-1):
                #     if angle_diff(angles[j], angles[j+1]) > threshold:
                #         print(f"lat = {lat_idx}")
                #         print(f"lon = {lon_idx}")
                #         print(f"a1 = {angles[j]}, a2 = {angles[j+1]}")
                #         print("Switch DETECTE ^^ \n")

                switches = sum(angle_diff(angles[t], angles[t+1]) > threshold for t in range(len(angles)-1))
                switches_count[lat_idx, lon_idx] = switches

        avg_switches = np.mean(switches_count)
        print(f"Average number of >{threshold}° switches per grid point in level {level_idx}: {avg_switches:.2f}/ {time_size}")

        total_switches = np.sum(switches_count)
        print(f"Total number of >{threshold}° switches for all in level {level_idx}: {total_switches}/ {lat_size*lon_size * time_size}")

        percentage_switches = avg_switches/total_count * 100
        print(f"% switches: {percentage_switches:.2f}%")
        # # Save to a plain text file
        output_dir = "Ocean/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"switches_count{level_idx}.txt")
        with open(output_path, "w") as f:
            f.write(f"In level {level_idx}\n")
            f.write(f"% switches: {percentage_switches:.2f}%\n")
            f.write(f"Average number of >{threshold}° switches per grid point in level {level_idx}: {avg_switches:.2f}/ {time_size}\n")
            f.write(f"Total number of >{threshold}° switches for all in level {level_idx}: {total_switches}/ {lat_size*lon_size * time_size}\n")

            for i in range(lat_size):
                for j in range(lon_size):
                    f.write(f"{switches_count[i, j]} ")
                    f.write("\t")
                f.write("\n")

        print(f"Saved switches_count to switches_count{level_idx}.txt")

    # Compute the average number of switches over all lat/lon points
   





if __name__ == '__main__':
    main()
# TODO: loop through lat

    # TODO: loop through long

        # TODO: for each point, extract the swdir variable in the 3 levels
        # TODO: extract the swdir variable in the 3 levels of the next time stamp

        # TODO: compare those 2 with my utility function to see if there is a switch in the time stamp
