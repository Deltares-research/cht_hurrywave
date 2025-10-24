import xarray as xr
import os
import glob
import pandas as pd
from pathlib import Path
import xarray as xr
import os
import glob
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
SRC_BASE = "/scratch-shared/dvdhoorn/ERA5_data_nonedit"
DST_NORTHSEA_BASE = "/scratch-shared/dvdhoorn/ERA5_data_NorthSea"
DST_STATION_BASE = "/scratch-shared/dvdhoorn/ERA5_data_per_station"
DST_COMBINED_BASE = "/scratch-shared/dvdhoorn/ERA5_data_per_station_combined"
OBS_FILE = "/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/08_Sims/templates/hurrywave.obs"

# North Sea bounding box
NORTH, WEST, SOUTH, EAST = 65, -12, 48, 10


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def read_stations(file_path):
    """Read station coordinates and names from a .obs file formatted as:
    lon lat # station_name
    """
    stations = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("#")
            coords = parts[0].strip().split()
            if len(coords) < 2:
                continue
            lon, lat = map(float, coords[:2])
            name = parts[1].strip().replace(" ", "_") if len(parts) > 1 else f"station_{len(stations)+1}"
            stations.append({"lon": lon, "lat": lat, "name": name})
    return pd.DataFrame(stations)


def crop_north_sea(ds):
    """Crop dataset to North Sea bounding box."""
    ds = ds.sel(latitude=slice(NORTH, SOUTH), longitude=slice(WEST, EAST))
    return ds


def extract_station_data(ds, stations):
    """Extract nearest grid point for each station."""
    results = {}
    for _, row in stations.iterrows():
        pt = ds.sel(latitude=row["lat"], longitude=row["lon"], method="nearest")
        results[row["name"]] = pt
    return results


def combine_station_files(variable_dir, output_path):
    """Combine all per-station files for a variable into one NetCDF."""
    station_files = sorted(glob.glob(f"{variable_dir}/*.nc"))
    if not station_files:
        print(f"⚠️ No station files found in {variable_dir}")
        return

    combined = []
    for sf in station_files:
        try:
            ds = xr.open_dataset(sf)

            # Normalize time coordinate name
            if "valid_time" in ds.coords and "time" not in ds.coords:
                ds = ds.rename({"valid_time": "time"})

            # Ensure consistent order and remove unexpected coords
            common_coords = [c for c in ds.coords if c in ["time", "latitude", "longitude"]]
            ds = ds.set_coords(common_coords)

            # Derive station name from filename
            station_name = os.path.basename(sf).split("_")[0]
            ds = ds.expand_dims(station=[station_name])
            combined.append(ds)

        except Exception as e:
            print(f"⚠️ Could not read {sf}: {e}")

    if combined:
        print(f"→ Combining {len(combined)} stations into {output_path}")
        merged = xr.concat(combined, dim="station", coords="minimal", compat="override")
        merged.to_netcdf(output_path)
        for ds in combined:
            ds.close()
        merged.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    # Ensure base output directories exist
    Path(DST_NORTHSEA_BASE).mkdir(parents=True, exist_ok=True)

    for variable in os.listdir(SRC_BASE):
        src_dir = os.path.join(SRC_BASE, variable)
        if not os.path.isdir(src_dir):
            continue

        out_dir_northsea = Path(DST_NORTHSEA_BASE) / variable
        out_dir_northsea.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing variable: {variable}")
        nc_files = sorted(glob.glob(f"{src_dir}/*.nc"))
        print(f"Found {len(nc_files)} files.")

        for nc in nc_files:
            fname = os.path.basename(nc)
            print(f"→ Cropping {fname}")
            try:
                ds = xr.open_dataset(nc)
                cropped = crop_north_sea(ds)
                cropped_path = out_dir_northsea / fname
                cropped.to_netcdf(cropped_path)

                ds.close()
                cropped.close()
            except Exception as e:
                print(f"⚠️ Error processing {fname}: {e}")

    print("\n✅ Processing complete.")


if __name__ == "__main__":
    main()