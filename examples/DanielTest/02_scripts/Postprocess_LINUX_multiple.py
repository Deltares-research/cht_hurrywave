import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs 
import cartopy.feature as cfeature
import xarray as xr
from scipy.spatial import cKDTree
import glob
import argparse
from datetime import datetime
from mpl_toolkits.basemap import Basemap

# --- Command-line interface ---
# parser = argparse.ArgumentParser(description="Postprocess Hurrywave output over multiple years.")
# parser.add_argument("--years", nargs='+', type=int, required=True, help="Years to concatenate and process")
# args = parser.parse_args()
years = [2019, 2020, 2021, 2022, 2023]  # Example years; replace with args.years if using CLI

# --- Paths ---
base_path = "/gpfs/work3/0/ai4nbs/hurry_wave/north_sea"
output_dir = os.path.join(base_path, '05_postprocessing', '5_years')
os.makedirs(output_dir, exist_ok=True)



# -------------------- Utility Functions --------------------
def open_and_concat_nc(files, concat_dim="time"):
    datasets = [xr.open_dataset(f, decode_times=True) for f in files]
    ds_concat = xr.concat(datasets, dim=concat_dim)
    for ds in datasets:
        ds.close()
    return ds_concat

def read_station_names_from_obs(file_path):
    names = []
    with open(file_path, 'r') as f:
        for line in f:
            if '#' in line:
                name = line.split('# ')[1].strip()
                names.append(name)
    return names

def parse_inp_file(inp_file):
    tstart = tstop = tspinup = None
    with open(inp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('tstart'):
                tstart = line.split('=')[1].strip()
            if line.strip().startswith('tstop'):
                tstop = line.split('=')[1].strip()
            if line.strip().startswith('tspinup'):
                tspinup = line.split('=')[1].strip()
    def parse_time(s):
        parts = s.strip().split()
        if len(parts) == 2:
            date, time = parts
        elif len(parts) == 1:
            date = parts[0][:8]
            time = parts[0][8:] if len(parts[0]) > 8 else "000000"
        else:
            date, time = "00000000", "000000"
        return f"{date} {time}"
    return parse_time(tstart), parse_time(tstop), tspinup

def remove_spinup(ds, tspinup, tstart_dt):
    if tspinup is None:
        return ds
    tspinup_seconds = int(float(tspinup))
    spinup_end_time = tstart_dt + pd.Timedelta(seconds=tspinup_seconds)
    time_values = pd.to_datetime(ds["time"].values)
    valid_indices = np.where(time_values >= spinup_end_time)[0]
    return ds.isel(time=valid_indices)

def extract_data(ds):
    nested_dict = {}
    for station in ds.stations.values:
        station_data = ds.sel(stations=station)
        nested_dict[station] = {}
        for var in ds.data_vars:
            nested_dict[station][var] = station_data[var].values
        for coord in ds.coords:
            nested_dict[station][coord] = station_data[coord].values
    return nested_dict

def load_and_crop(ds, varname, tstart_dt, tstop_dt):
    if 'valid_time' not in ds.coords:
        if 'time' in ds.coords:
            ds = ds.rename({'time': 'valid_time'})
        else:
            raise ValueError("No 'valid_time' or 'time' coordinate found in dataset.")
    ds = ds.sel(valid_time=slice(tstart_dt, tstop_dt))
    lat_name = [k for k in ds.coords if 'lat' in k.lower()][0]
    lon_name = [k for k in ds.coords if 'lon' in k.lower()][0]
    if ds[lon_name].max() > 180:
        ds[lon_name] = ((ds[lon_name] + 180) % 360) - 180
        ds = ds.sortby(lon_name)
    return ds[[varname]]

def extract_and_merge_buoy_data(paths):
    all_data = {}
    for path in paths:
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        for csv_file in csv_files:
            station = os.path.splitext(os.path.basename(csv_file))[0]
            df = pd.read_csv(csv_file)
            if station not in all_data:
                all_data[station] = {}
            for col in df.columns:
                if col not in all_data[station]:
                    all_data[station][col] = list(df[col].values)
                else:
                    # Ensure we are always working with lists before extending
                    if isinstance(all_data[station][col], np.ndarray):
                        all_data[station][col] = all_data[station][col].tolist()
                    all_data[station][col].extend(list(df[col].values))
    # After merging, convert columns to arrays and process time/hm0
    for station in all_data:
        if 'Unnamed: 0' in all_data[station]:
            time_pd = pd.to_datetime(all_data[station].pop('Unnamed: 0'))
            all_data[station]['time'] = time_pd.values.astype('datetime64[ns]')
        for col in all_data[station]:
            all_data[station][col] = np.array(all_data[station][col])
        if 'hm0' in all_data[station]:
            all_data[station]['hm0'] = all_data[station]['hm0'] / 100.0
    return all_data

def find_valid_wave_points(ds, lats, lons, wave_var="swh"):
    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    result_lats, result_lons = [], []
    for lat, lon in zip(lats, lons):
        lat_idx = np.abs(lat_vals - lat).argmin()
        lon_idx = np.abs(lon_vals - lon).argmin()
        result_lats.append(lat_vals[lat_idx])
        result_lons.append(lon_vals[lon_idx])
    return result_lats, result_lons

def extract_data_era5(ds, latitudes, longitudes, station_names):
    station_data = []
    for name, lat, lon in zip(station_names, latitudes, longitudes):
        point_data = ds.sel(latitude=lat, longitude=lon, method='nearest')
        point_data = point_data.expand_dims(dim='stations')
        point_data = point_data.assign_coords(stations=[name])
        station_data.append(point_data)
    return xr.concat(station_data, dim='stations')

def synchronize_times(df_era5=None, df_model=None, df_measurements=None, station_names=None, sync_with=("model", "era5", "measurements")):
    """
    Synchronize time series across provided dataframes for each station.
    sync_with: tuple/list of which dataframes to synchronize with. Options: "model", "era5", "measurements".
    Only the provided dataframes and those listed in sync_with will be synchronized.
    """
    # Build a dict of available dataframes
    dfs = {}
    if df_model is not None and "model" in sync_with:
        dfs["model"] = df_model
    if df_era5 is not None and "era5" in sync_with:
        dfs["era5"] = df_era5
    if df_measurements is not None and "measurements" in sync_with:
        dfs["measurements"] = df_measurements

    # Determine the time key for each dataframe
    time_keys = {
        "model": "time",
        "era5": "valid_time",
        "measurements": "time"
    }

    for station in station_names:
        # Collect all time arrays for this station
        time_sets = []
        for key, df in dfs.items():
            if station in df and time_keys[key] in df[station]:
                time_sets.append(set(df[station][time_keys[key]]))
        if not time_sets:
            continue
        # Find intersection of all available times
        common_times = set.intersection(*time_sets)
        # Synchronize each dataframe to common_times
        for key, df in dfs.items():
            if station not in df or time_keys[key] not in df[station]:
                continue
            time_arr = df[station][time_keys[key]]
            mask = np.in1d(time_arr, list(common_times))
            for var in df[station]:
                arr = df[station][var]
                if hasattr(arr, 'shape') and arr.shape[0] == len(time_arr):
                    df[station][var] = arr[mask]

def compute_benchmark_statistics(model_dict, model_vars, era5_dict=None, buoy_dict=None, benchmarks=["era5", "buoy"], quantile_pinball=[], map_variable_era5=None, map_variable_buoy=None):
    def rmse(obs, pred): return np.sqrt(np.mean((np.array(pred) - np.array(obs)) ** 2))
    def bias(obs, pred): return np.mean(np.array(pred) - np.array(obs))
    def scatter_index(obs, pred): return rmse(obs, pred) / np.mean(np.array(obs))
    def pinball_loss(obs, pred, q):
        delta = np.array(obs) - np.array(pred)
        return np.mean(np.maximum(q * delta, (q - 1) * delta))
    df_stats = {}
    for station in model_dict:
        df_stats[station] = {}
        for model_var in model_vars:
            model_values = model_dict[station].get(model_var)
            if model_values is None:
                continue
            df_stats[station][model_var] = {}
            if "era5" in benchmarks and era5_dict:
                obs = era5_dict[station].get(map_variable_era5(model_var))
                if obs is not None:
                    df_stats[station][model_var]["RMSE"] = {"era5": rmse(obs, model_values)}
                    df_stats[station][model_var]["Bias"] = {"era5": bias(obs, model_values)}
                    df_stats[station][model_var]["Scatter Index"] = {"era5": scatter_index(obs, model_values)}
                    for q in quantile_pinball:
                        df_stats[station][model_var][f"Pinball Loss (q={q})"] = {"era5": pinball_loss(obs, model_values, q)}
            if "buoy" in benchmarks and buoy_dict:
                obs = buoy_dict[station].get(map_variable_buoy(model_var))
                if obs is not None:
                    df_stats[station][model_var].setdefault("RMSE", {})["buoy"] = rmse(obs, model_values)
                    df_stats[station][model_var].setdefault("Bias", {})["buoy"] = bias(obs, model_values)
                    df_stats[station][model_var].setdefault("Scatter Index", {})["buoy"] = scatter_index(obs, model_values)
                    for q in quantile_pinball:
                        df_stats[station][model_var].setdefault(f"Pinball Loss (q={q})", {})["buoy"] = pinball_loss(obs, model_values, q)
    return df_stats

def plot_station_data_comparison(
    station_name,
    model_df,
    era5_df=None,
    buoy_df=None,
    model_vars=None,
    benchmarks=["era5", "buoy"],
    map_variable_era5=None,
    map_variable_buoy=None,
    df_statistics=None,
    output_dir=output_dir
):
    """
    Plot time series for selected variables at a single station, comparing model, ERA5, and/or buoy data.
    Includes a map with the station location and a side panel showing statistical metrics.

    Parameters:
        station_name (str): Name of the station to plot.
        model_df (dict): Nested dict model[station][variable] = values.
        era5_df (dict): Same structure as model_df for ERA5 (optional).
        buoy_df (dict): Same structure as model_df for buoy data (optional).
        model_vars (list): Model variable names to include in the plot.
        benchmarks (list): Which benchmarks to include ("era5", "buoy", or both).
        variable_mapping_era5 (dict): Mapping from model vars to ERA5 vars.
        variable_mapping_buoy (dict): Mapping from model vars to buoy vars.
        df_statistics (dict): Nested dict of statistics.
    """
    if model_vars is None:
        model_vars = list(model_df.get(station_name, {}).keys())

    n_vars = len(model_vars)
    plot_height = 0.18
    plot_space = 0.04  # space between plots

    fig_height = 0.8 + n_vars * (plot_height + plot_space)
    fig = plt.figure(figsize=(14, 3 * n_vars + 5))
    fig.suptitle(f'Data Comparison at {station_name}', fontsize=16)

    # Adjust map position on the figure
    map_ax = fig.add_axes([0.05, 0.8, 0.6, 0.15], projection=ccrs.Mercator())
    map_ax.set_extent([-5, 10, 50, 65], crs=ccrs.PlateCarree())

    # Add coastlines and filled continents
    map_ax.coastlines(resolution='50m')
    map_ax.add_feature(cfeature.LAND, facecolor='lightgray')
    map_ax.add_feature(cfeature.OCEAN, facecolor='aqua')
    map_ax.add_feature(cfeature.LAKES, facecolor='aqua')

    lat = float(model_df[station_name]["station_y"])
    lon = float(model_df[station_name]["station_x"])
    map_ax.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())
    map_ax.text(lon, lat, f' {station_name}', fontsize=10, color='black', transform=ccrs.PlateCarree())

    # Adjust time series plots position
    axs = []
    stat_axs = []
    for i in range(n_vars):
        bottom = 0.6 - i * (plot_height + plot_space)
        ax = fig.add_axes([0.05, bottom, 0.6, plot_height])
        axs.append(ax)
        stat_ax = fig.add_axes([0.7, bottom, 0.25, plot_height])
        stat_axs.append(stat_ax)

    time = model_df[station_name]["time"]

    for i, var in enumerate(model_vars):
        ax = axs[i]
        stat_ax = stat_axs[i]
        ax.grid(True)

        # Model
        if var in model_df[station_name]:
            ax.plot(time, model_df[station_name][var], label="Model", color='blue')

        # ERA5
        if "era5" in benchmarks and era5_df:
            era5_var = map_variable_era5.get(var, var) if map_variable_era5 else var
            if era5_var in era5_df.get(station_name, {}):
                ax.plot(time, era5_df[station_name][era5_var], label="ERA5", linestyle='--', color='green')

        # Buoy
        if "buoy" in benchmarks and buoy_df:
            buoy_var = map_variable_buoy.get(var, var) if map_variable_buoy else var
            if buoy_var in buoy_df.get(station_name, {}):
                ax.plot(time, buoy_df[station_name][buoy_var], label="Buoy", linestyle=':', color='orange')

        ax.set_title(var)
        ax.set_ylabel(var)
        ax.legend()

        # Display statistics
        stat_ax.axis('off')
        if df_statistics and station_name in df_statistics and var in df_statistics[station_name]:
            lines = []
            stat_data = df_statistics[station_name][var]
            for stat in stat_data:
                line = f"{stat.capitalize()}"
                for source in benchmarks:
                    if source in stat_data[stat]:
                        val = stat_data[stat][source]
                        line += f" ({source}): {val:.3f} ;"
                line = line.rstrip(" ;")
                lines.append(line)
            stat_ax.text(0, 1, '\n'.join(lines), fontsize=9, va='top')

    axs[-1].set_xlabel("Time")
    output_path = os.path.join(output_dir, f'His_graph_at_{station_name}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

def plot_all_stations(
    station_names,
    model_df,
    era5_df=None,
    buoy_df=None,
    model_vars=None,
    benchmarks=["era5", "buoy"],
    map_variable_era5=None,
    map_variable_buoy=None,
    df_statistics=None,
    output_dir=output_dir):

    """
    Plot time series for selected variables at all stations, comparing model, ERA5, and/or buoy data.
    Includes a map with the station locations and a side panel showing statistical metrics.
    Parameters:
        station_names (list of str): Names of the station to plot.
        model_df (dict): Nested dict model[station][variable] = values.
        era5_df (dict): Same structure as model_df for ERA5 (optional).
        buoy_df (dict): Same structure as model_df for buoy data (optional).
        model_vars (list): Model variable names to include in the plot.
        benchmarks (list): Which benchmarks to include ("era5", "buoy", or both).
        variable_mapping_era5 (dict): Mapping from model vars to ERA5 vars.
        variable_mapping_buoy (dict): Mapping from model vars to buoy vars.
        df_statistics (dict): Nested dict of statistics.
    """
    
    for station_name in station_names:
        plot_station_data_comparison(
            station_name,
            model_df,
            era5_df=era5_df,
            buoy_df=buoy_df,
            model_vars=model_vars,
            benchmarks=benchmarks,
            map_variable_era5=map_variable_era5,
            map_variable_buoy=map_variable_buoy,
            df_statistics=df_statistics,
            output_dir=output_dir
        )

# SCATTER PLOTS
def scatter_plot_station_data_comparison(
    station_name,
    model_df,
    era5_df=None,
    buoy_df=None,
    model_vars=None,
    benchmarks=["era5", "buoy"],
    map_variable_era5=None,
    map_variable_buoy=None,
    df_statistics=None,
    output_dir=None
):  
    """
    Plot map and scatter comparison between model, ERA5, and buoy at a station.
    """
    if model_vars is None:
        model_vars = list(model_df.get(station_name, {}).keys())

    fig = plt.figure(figsize=(14, 2.5 + len(model_vars) * 2.5))
    fig.suptitle(f'Data Comparison at {station_name}', fontsize=18, y=0.97, ha='center')

    # Larger map on top, centered
    map_ax = fig.add_axes([0.08, 0.72, 0.84, 0.22])
    m = Basemap(projection='merc', llcrnrlat=50, urcrnrlat=65,
                llcrnrlon=-5, urcrnrlon=10, resolution='i', ax=map_ax)
    m.drawcoastlines()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')

    lat = float(model_df[station_name]["station_y"])
    lon = float(model_df[station_name]["station_x"])
    x, y = m(lon, lat)
    m.plot(x, y, 'ro', markersize=8)
    map_ax.text(x, y, f' {station_name}', fontsize=10, color='black')

    scatter_vars = model_vars
    n_rows = len(scatter_vars)

    top_offset = 0.72
    plot_height = 0.5
    vertical_space = 0.1
    plot_width = 0.5
    hspace = 0.0
    left1 = 0.1
    left2 = left1 + plot_width + hspace

    for row_idx, var in enumerate(scatter_vars):
        model_vals = model_df[station_name].get(var)
        if model_vals is None:
            continue

        era5_var = map_variable_era5.get(var, var) if map_variable_era5 else var
        buoy_var = map_variable_buoy.get(var, var) if map_variable_buoy else var
        bottom = top_offset - (row_idx + 1) * (plot_height + vertical_space)

        # Model vs ERA5
        if "era5" in benchmarks and era5_df and era5_var in era5_df.get(station_name, {}):
            rmse = df_statistics[station_name][var]['RMSE']['era5'] if df_statistics and station_name in df_statistics and var in df_statistics[station_name] else np.nan
            era5_vals = era5_df[station_name][era5_var]
            ax1 = fig.add_axes([left1, bottom, plot_width, plot_height])
            ax1.scatter(model_vals, era5_vals, s=10, alpha=0.6, color='green')
            min_val = min(np.nanmin(model_vals), np.nanmin(era5_vals))
            max_val = max(np.nanmax(model_vals), np.nanmax(era5_vals))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--')
            ax1.set_xlim(min_val, max_val)
            ax1.set_ylim(min_val, max_val)
            ax1.set_aspect('equal', adjustable='box')
            # Print RMSE in top left of the scatter plot
            ax1.text(
                0.02, 0.98,
                f"RMSE: {rmse:.3f}",
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            ax1.set_xlabel("Model")
            ax1.set_ylabel("ERA5")
            ax1.set_title(f"{var} - Model vs ERA5")
            ax1.grid(True)


        # Model vs Buoy
        if "buoy" in benchmarks and buoy_df and buoy_var in buoy_df.get(station_name, {}):
            rmse = df_statistics[station_name][var]['RMSE']['buoy'] if df_statistics and station_name in df_statistics and var in df_statistics[station_name] else np.nan
            buoy_vals = buoy_df[station_name][buoy_var]
            ax2 = fig.add_axes([left2, bottom, plot_width, plot_height])
            ax2.scatter(model_vals, buoy_vals, s=10, alpha=0.6, color='orange')
            min_val = min(np.nanmin(model_vals), np.nanmin(buoy_vals))
            max_val = max(np.nanmax(model_vals), np.nanmax(buoy_vals))
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--')
            ax2.set_xlim(min_val, max_val)
            ax2.set_ylim(min_val, max_val)
            ax2.set_aspect('equal', adjustable='box')
            # Print RMSE in top left of the scatter plot
            ax2.text(
                0.02, 0.98,
                f"RMSE: {rmse:.3f}",
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            ax2.set_xlabel("Model")
            ax2.set_ylabel("Buoy")
            ax2.set_title(f"{var} - Model vs Buoy")
            ax2.grid(True)

    output_path = os.path.join(output_dir, f'Scatter_comparison_at_{station_name}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def scatter_plot_all_stations(
    station_names,
    model_df,
    era5_df=None,
    buoy_df=None,
    model_vars=None,
    benchmarks=["era5", "buoy"],
    map_variable_era5=None,
    map_variable_buoy=None,
    df_statistics=None,
    output_dir=output_dir):

    """
    Plot time series for selected variables at all stations, comparing model, ERA5, and/or buoy data.
    Includes a map with the station locations and a side panel showing statistical metrics.
    Parameters:
        station_names (list of str): Names of the station to plot.
        model_df (dict): Nested dict model[station][variable] = values.
        era5_df (dict): Same structure as model_df for ERA5 (optional).
        buoy_df (dict): Same structure as model_df for buoy data (optional).
        model_vars (list): Model variable names to include in the plot.
        benchmarks (list): Which benchmarks to include ("era5", "buoy", or both).
        variable_mapping_era5 (dict): Mapping from model vars to ERA5 vars.
        variable_mapping_buoy (dict): Mapping from model vars to buoy vars.
        df_statistics (dict): Nested dict of statistics.
    """
    
    for station_name in station_names:
        scatter_plot_station_data_comparison(
            station_name,
            model_df,
            era5_df=era5_df,
            buoy_df=buoy_df,
            model_vars=model_vars,
            benchmarks=benchmarks,
            map_variable_era5=map_variable_era5,
            map_variable_buoy=map_variable_buoy,
            df_statistics=df_statistics,
            output_dir=output_dir
        )


# --- Gather model data ---
his_files = [
    os.path.join(base_path, '04_modelruns', 'YearSims', str(year), 'hurrywave_his.nc')
    for year in years
]
his = open_and_concat_nc(his_files)

print(f"Loaded his")

# --- Extract station names and spinup info from first year's input ---
first_year = str(years[0])
model_path = os.path.join(base_path, '04_modelruns', 'YearSims', first_year)
inp_file = os.path.join(model_path, 'hurrywave.inp')
obs_file_path = os.path.join(model_path, 'hurrywave.obs')
station_names = read_station_names_from_obs(obs_file_path)

# Extract tstart, tstop, tspinup
tstart, tstop, tspinup = parse_inp_file(inp_file)
print(tstart, tstop, tspinup)
tstart_dt = datetime.strptime(tstart, "%Y%m%d %H%M%S")
tstop_dt = datetime.strptime(tstop, "%Y%m%d %H%M%S")
his = remove_spinup(his, tspinup, tstart_dt)
print(f"Spinup removed until: {tstart_dt + pd.Timedelta(seconds=int(float(tspinup)))}")
his = his.assign_coords({"stations": station_names})
df_model = extract_data(his)

print(f"Loaded model data for years: {years}")

# --- Load and merge ERA5 ---
# ERA5_data_path = "/gpfs/work3/0/ai4nbs/ERA5_data"
# ERA5_netcdf_files = {
#     'swh': [os.path.join(ERA5_data_path, 'significant_height_of_combined_wind_waves_and_swell', f'global_significant_height_of_combined_wind_waves_and_swell_{year}.nc') for year in years],
#     'mwp': [os.path.join(ERA5_data_path, 'mean_wave_period', f'global_mean_wave_period_{year}.nc') for year in years],
# }

# datasets = []
# for varname, filelist in ERA5_netcdf_files.items():
#     ds_concat = open_and_concat_nc(filelist)
#     ds_cropped = load_and_crop(ds_concat, varname, tstart_dt, tstop_dt)
#     datasets.append(ds_cropped)

# print(f"Loaded and cropped ERA5 data for variables: {list(ERA5_netcdf_files.keys())}")
# era5_data = xr.merge(datasets)

# print(f"ERA5 data merged")
# --- Load and merge buoy data ---
buoy_data_path_list = [os.path.join(base_path, '01_data/Waterinfo_RWS', str(y)) for y in years]
df_measurements = extract_and_merge_buoy_data(buoy_data_path_list)

print(f"Loaded buoy data for years: {years}")
# --- Extract ERA5 values at station points ---
# x = [float(df_model[station]['station_x']) for station in df_model]
# y = [float(df_model[station]['station_y']) for station in df_model]
# station_names = list(df_model.keys())
# valid_latitudes, valid_longitudes = find_valid_wave_points(era5_data, y, x)
# era5_data_stations = extract_data_era5(era5_data, valid_latitudes, valid_longitudes, station_names)
# df_era5 = extract_data(era5_data_stations)

# --- Synchronize time series ---
synchronize_times(df_era5=None, df_model=df_model, df_measurements=df_measurements, station_names=station_names, sync_with=("model",  "measurements"))

print(f"Synchronized time series for stations: {station_names}")
# --- Define variables and mappings ---
variable_mapping_era5 = {'point_hm0': 'swh', 'point_tp': 'mwp'}
variable_mapping_measurements = {'point_hm0': 'hm0', 'point_tp': 't13'}
variable_list = list(variable_mapping_era5.keys())
pinball_quantiles = [0.05,0.1, 0.25, 0.5, 0.75, 0.9,0.95]

# --- Compute statistics ---
df_statistics = compute_benchmark_statistics(
    df_model,
    variable_list,
    buoy_dict=df_measurements, 
    quantile_pinball=pinball_quantiles,
    benchmarks=[ "buoy"],
    map_variable_era5=lambda v: variable_mapping_era5.get(v, v),
    map_variable_buoy=lambda v: variable_mapping_measurements.get(v, v)
)

print(f"Computed statistics for benchmarks: {list(df_statistics.keys())}")
# --- Generate plots ---
plot_all_stations(
    station_names=station_names,
    model_df=df_model,
    buoy_df=df_measurements,
    model_vars=variable_list,
    benchmarks=["buoy"],
    map_variable_era5=variable_mapping_era5,
    map_variable_buoy=variable_mapping_measurements,
    df_statistics=df_statistics,
    output_dir=output_dir
)

scatter_plot_all_stations(
    station_names=station_names,
    model_df=df_model,
    buoy_df=df_measurements,
    model_vars=variable_list,
    benchmarks=["buoy"],
    map_variable_era5=variable_mapping_era5,
    map_variable_buoy=variable_mapping_measurements,
    df_statistics=df_statistics,
    output_dir=output_dir
)