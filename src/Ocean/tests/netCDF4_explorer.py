from netCDF4 import Dataset
import numpy as np

lat_target = 55.54
lon_target = 225

# 1. Open .nc files
ds = Dataset(
    "Ocean/tests/weather_2_nc/0101_0110_lat-5_5_lon270_280_fc50_ts9/gfsWave_Global25_20250101_00_000.nc",
    AccessMode="r",
)

# 2. explore data
print(ds)

# Dimensions
print("\nDimensions:")

# Variables
print("\nVariables:")
print(ds.variables.keys())

for name, var in ds.variables.items():
    print(
        f"  {name}: shape = {var.shape}, dtype = {var.datatype}, unit = {getattr(var, 'units', 'N/A')}"
    )
    for attr in var.ncattrs():
        print(f"    {attr} = {getattr(var, attr)}")

    # Global Attributes
print("\nGlobal Attributes:")
for attr in ds.ncattrs():
    print(f"    {attr}: {getattr(ds, attr)}")

# 3. Read variable
varname = "mpts"
vardata = ds.variables[varname]

print(f"\n Exploring variable {varname}:")
print(vardata)

# 4. Read Coordinates
lats = ds.variables["lat"][:]
lons = ds.variables["lon"][:]

lat_idx = np.abs(lats - lat_target).argmin()
lon_idx = np.abs(lons - lon_target).argmin()


print(f"\nExtract {varname} closest to {lat_target}°N and {lon_target}°E")
print("Closest lat:", lats[lat_idx])
print("Closest lon:", lons[lon_idx])
value = vardata[lat_idx, lon_idx]
print(varname, "value: ", value, ds.variables[varname].units)

# 5. Close the dataset
ds.close()
