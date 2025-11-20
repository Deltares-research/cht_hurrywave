import os
import xarray as xr
import numpy as np

def extract_stations_from_global(
    input_dir,
    output_dir,
    station_file,
    area=[65, -12, 48, 10]
):
    """
    Create per-year, per-variable station-only ERA5 files
    from existing North Sea ERA5 files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load station info
    stations = []
    with open(station_file, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split("#")
                lonlat = parts[0].split()
                name = parts[1].strip().replace(" ", "_") if len(parts) > 1 else f"station_{len(stations)+1}"
                lon, lat = map(float, lonlat)
                stations.append((name, lon, lat))

    print(f"Loaded {len(stations)} stations from {station_file}")
    print("Stations:", [s[0] for s in stations])

    # Loop through variables (subfolders)
    for var_folder in sorted(os.listdir(input_dir)):
        var_path = os.path.join(input_dir, var_folder)
        if not os.path.isdir(var_path):
            continue

        output_var_dir = os.path.join(output_dir, var_folder)
        os.makedirs(output_var_dir, exist_ok=True)

        for fname in sorted(os.listdir(var_path)):
            if not fname.endswith(".nc"):
                continue

            in_path = os.path.join(var_path, fname)
            fname_out = fname.replace("global", "node")
            out_path = os.path.join(output_var_dir, fname_out)

            print(f"\n→ Processing {in_path}")

            ds = xr.open_dataset(in_path)

            # Detect coordinate names
            lat_name = [c for c in ds.coords if "lat" in c.lower()][0]
            lon_name = [c for c in ds.coords if "lon" in c.lower()][0]
            time_name = [c for c in ds.coords if "time" in c.lower()][0]

            # Normalize longitudes
            if ds[lon_name].max() > 180:
                ds[lon_name] = ((ds[lon_name] + 180) % 360) - 180
                ds = ds.sortby(lon_name)

            # Crop to area
            ds = ds.sel({
                lat_name: slice(area[0], area[2]),
                lon_name: slice(area[1], area[3])
            })

            # Extract station data
            station_data_list = []
            lons, lats, names = [], [], []

            for name, lon, lat in stations:
                subset = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
                nearest_lon = float(subset[lon_name])
                nearest_lat = float(subset[lat_name])
                print(f"  {name:30s} → nearest gridpoint ({nearest_lon:.3f}, {nearest_lat:.3f})")

                lons.append(nearest_lon)
                lats.append(nearest_lat)
                names.append(name)

                # Add station dimension
                subset = subset.expand_dims(dim={"station": [name]})
                station_data_list.append(subset)

            # Combine all stations along the new station dimension
            combined = xr.concat(station_data_list, dim="station")

            # Drop original lat/lon coords and assign station coordinates
            combined = combined.drop_vars([lat_name, lon_name])
            combined = combined.assign_coords(
                longitude=("station", lons),
                latitude=("station", lats)
            )

            # Save output
            combined.to_netcdf(out_path)
            print(f"  ✓ Saved {out_path} ({len(names)} stations)")

            ds.close()

    print("\nAll done!")


if __name__ == "__main__":
    # Example paths
    ERA5_global_dir = "/scratch-shared/dvdhoorn/ERA5_data_NorthSea"
    ERA5_stations_dir = "/scratch-shared/dvdhoorn/ERA5_data_per_station_combined"
    station_file = "/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/08_Sims/templates/hurrywave.obs"

    extract_stations_from_global(
        input_dir=ERA5_global_dir,
        output_dir=ERA5_stations_dir,
        station_file=station_file
    )