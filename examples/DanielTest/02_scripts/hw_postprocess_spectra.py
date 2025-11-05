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
from mpl_toolkits.basemap import Basemap
import scipy.stats as stats
import netCDF4 as nc
import wavespectra
import cmocean
from datetime import datetime, timedelta
import scipy.io

# base_path = '/home/dvdhoorn/hurrywave_runs'
base_path = '/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/04_modelruns/YearSims'

year_start = 2009
year_end = 2015
# model_name = '2013'
model_names = [str(year) for year in range(year_start, year_end + 1)]
model_paths = [os.path.join(base_path, model_name) for model_name in model_names]

# nc_file = os.path.join(model_path,'hurrywave_sp2.nc')
# inp_file = os.path.join(model_path,'hurrywave.inp')

# if os.path.exists(os.path.join(model_path, 'hurrywave.log')):
#     log_file = os.path.join(model_path, 'hurrywave.log')
# else:
#     log_file = None
#     print('WARNING: Log file not found at:', log_file)

postprocess_path = '/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/05_postprocessing/Spectra'
filepath_data = '/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/01_data/spectral_buoy_data'
timestep = 500
lat_A12 = 55.399167
lon_A12 = 3.810278

def get_dt(inp_file, log_file):
    dt = None
    if log_file and os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if 'Warning! Reducing dt from' in line:
                    parts = line.strip().split('to')
                    if len(parts) == 2:
                        try:
                            dt = float(parts[1].split()[0])
                            print('taking dt from log file')
                            print(f'dt found in log file: {dt} s')
                            return dt
                        except Exception:
                            pass
    # If not found in log file, read from input file
    with open(inp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('dt'):
                parts = line.split('=')
                if len(parts) == 2:
                    try:
                        dt = float(parts[1].split(',')[0].strip())
                        print('taking dt from input file')
                        print(f'dt found in input file: {dt} s')
                        return dt
                    except Exception:
                        pass
    raise ValueError("dt could not be found in log or input file")

def get_spectra_output_dt(inp_file):
    dtsp2out = None
    with open(inp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('dtsp2out'):
                parts = line.split('=')
                if len(parts) == 2:
                    try:
                        dtsp2out = float(parts[1].split(',')[0].strip())
                        print(f'spectra output dt found in input file: {dtsp2out} s')
                        return dtsp2out
                    except Exception:
                        pass
    print("WARNING: dtsp2out not found in input file")
    return None

def get_timesteps(inp_file, log_file):
    # Read tstart and tend from inp_file
    tstart = None
    tend = None
    with open(inp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('tstart'):
                tstart_str = line.split('=')[1].strip().split(',')[0]
                tstart = datetime.strptime(tstart_str, "%Y%m%d %H%M%S")
            if line.strip().startswith('tstop'):
                tstop_str = line.split('=')[1].strip().split(',')[0]
                tstop = datetime.strptime(tstop_str, "%Y%m%d %H%M%S")
    if tstart is None or tstop is None:
        raise ValueError("tstart or tend not found in inp file")

    dt = get_spectra_output_dt(inp_file)
    nsteps = int((tstop - tstart).total_seconds() // dt) + 1

    # Remove the spinup times
    t_spinup = None
    with open(inp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('tspinup'):
                t_spinup_str = line.split('=')[1].strip().split(',')[0]
                t_spinup = float(t_spinup_str)
                break
    if t_spinup is None:
        raise ValueError("tspinup not found in input file")
    

    time_steps = np.array([tstart + timedelta(seconds=i*dt) for i in range(nsteps)])

    # Remove the spinup times
    # Remove all time steps before tstart + tspinup seconds
    spinup_end_time = tstart + timedelta(seconds=t_spinup)
    time_steps = time_steps[time_steps >= spinup_end_time]

    return time_steps

for model_name in model_names:
    model_path = os.path.join(base_path, model_name)
    inp_file = os.path.join(model_path, 'hurrywave.inp')
    log_file = os.path.join(model_path, 'hurrywave.log') if os.path.exists(os.path.join(model_path, 'hurrywave.log')) else None

    this_model_timesteps = get_timesteps(inp_file, log_file)
    model_timesteps = np.concatenate((model_timesteps, this_model_timesteps)) if 'model_timesteps' in locals() else this_model_timesteps # concatenate all timesteps

    print("First timestep:", model_timesteps[0])
    print("Last timestep:", model_timesteps[-1])

def read_stations_from_obs(file_path):
    """
    Reads station coordinates and names from a .obs file.

    Returns:
        xs (list[float]): x coordinates
        ys (list[float]): y coordinates
        names (list[str]): station names
    """
    xs, ys, names = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            if '#' in line:
                parts = line.strip().split()
                x = float(parts[0])
                y = float(parts[1])
                name = " ".join(parts[3:])  # everything after "#"
                xs.append(x)
                ys.append(y)
                names.append(name)
    return xs, ys, names

# Example usage
obs_file_path = os.path.join('/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/08_Sims/templates', 'hurrywave.obs')
xs, ys, station_names = read_stations_from_obs(obs_file_path)

def read_stations_from_obs(file_path):
    xs, ys, names = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            if '#' in line:
                parts = line.strip().split()
                x = float(parts[0])
                y = float(parts[1])
                name = " ".join(parts[3:])
                xs.append(x)
                ys.append(y)
                names.append(name)
    return xs, ys, names

def his_remove_spinup(year, model_base_path, his):
    # Find the start and stop times from the input file
    inp_file = os.path.join(model_base_path, str(year), 'hurrywave.inp')
    with open(inp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('tspinup'):
                tspinup = line.split('=')[1].strip()

    tspinup_sec = int(tspinup)
    his_time_vals = his["time"].values
    spinup_end_time = his_time_vals[0] + np.timedelta64(tspinup_sec, 's')
    his_spinup_mask = his_time_vals >= spinup_end_time
    his = his.sel(time=his_spinup_mask)
    return his

def process_station_his_data(model_base_path, outdir, y0, y1, save=True):
    print('Starting Hurrywave station extraction...')

    years = np.arange(y0, y1 + 1)
    data_per_station = {}

    for y in years:
        print(f'Processing year: {y}')
        # Read station coordinates from obs file
        obs_file_path = os.path.join(model_base_path,str(y), 'hurrywave.obs')
        xs, ys, station_names_obs = read_stations_from_obs(obs_file_path)
        print(station_names_obs)
        station_coord_map = {name: (x, y) for name, x, y in zip(station_names_obs, xs, ys)}
        fname = os.path.join(model_base_path, str(y), 'hurrywave_his.nc')
        ds = xr.open_dataset(fname)

        ds = his_remove_spinup(y, model_base_path, ds)

        nstations = ds.dims['stations']
        station_names = ds['station_name'].astype(str).values

        # Select only point variables with dims (time, stations)
        data_vars = [v for v in ds.data_vars if ds[v].dims == ('time', 'stations')]
        time = ds['time'].values

        for i in range(nstations):
            station = station_names[i].strip()
            print(f'  - Extracting {station}')

            if station not in data_per_station:
                data_per_station[station] = {var: [] for var in data_vars}
                data_per_station[station]['time'] = []

                # Add station coordinates
                data_per_station[station]['station_x'] = station_coord_map[station_names_obs[i]][0]
                data_per_station[station]['station_y'] = station_coord_map[station_names_obs[i]][1]

            # Append year’s data for each variable
            for var in data_vars:
                data_per_station[station][var].append(ds[var].isel(stations=i).values)
            data_per_station[station]['time'].append(time)

        ds.close()

    # Concatenate lists into arrays per station
    for station in data_per_station.keys():
        for key in data_per_station[station]:
            if isinstance(data_per_station[station][key], list) and key not in ('station_x', 'station_y'):
                data_per_station[station][key] = np.concatenate(data_per_station[station][key])

    if save:
        print('All years loaded. Starting export...')
        os.makedirs(outdir, exist_ok=True)

        for station, data in data_per_station.items():
            outpath = os.path.join(outdir, f'{station}.npz')
            np.savez(outpath, **data)
            print(f'  ✓ Saved {station}.npz with {len(data["time"])} timesteps.')

        print(f'Extraction finished. Data saved to: {outdir}')
    else:
        print(f'Extraction complete. Data not saved (save=False).')

    return data_per_station, station_names_obs

data_per_station, station_names = process_station_his_data(
    model_base_path=base_path,
    outdir=postprocess_path,
    y0=year_start,
    y1=year_end,
    save=False
)

A12_dep = data_per_station['station_008']['point_depth']
Hoorn_dep = data_per_station['station_003']['point_depth']
EPL_dep = data_per_station['station_001']['point_depth']

def ds_remove_spinup(year, model_base_path, ds):
    # Find the start and stop times from the input file
    inp_file = os.path.join(model_base_path, str(year), 'hurrywave.inp')
    with open(inp_file, 'r') as f:
        for line in f:
            if line.strip().startswith('tspinup'):
                tspinup = line.split('=')[1].strip()

    tspinup_sec = int(tspinup)
    ds_time_vals = ds["time"].values
    spinup_end_time = ds_time_vals[0] + np.timedelta64(tspinup_sec, 's')
    ds_spinup_mask = ds_time_vals >= spinup_end_time
    ds = ds.sel(time=ds_spinup_mask)
    return ds

ds = xr.open_dataset('/home/dvdhoorn/hurrywave_runs/2013/hurrywave_sp2.nc')

def process_station_data(model_base_path, outdir, y0, y1, save=True):
    print('Starting Hurrywave station extraction...')

    years = np.arange(y0, y1 + 1)
    data_per_station = {}
    sigma = None
    theta = None

    for y in years:
        print(f'Processing year: {y}')
        fname = os.path.join(model_base_path, str(y), 'hurrywave_sp2.nc')
        ds = xr.open_dataset(fname)
        ds = ds_remove_spinup(y, model_base_path, ds)


        nstations = ds.dims['stations']
        station_names = ds['station_name'].astype(str).values
        if sigma is None:
            sigma = ds['sigma'].values
        if theta is None:
            theta = ds['theta'].values
            theta = (270 - theta) % 360 # Convert to meteorological convention

        time = ds['time'].values
        data_vars = ['point_spectrum2d']

        for i in range(nstations):
            station = station_names[i].strip()
            print(f'  - Extracting {station}')

            if station not in data_per_station:
                data_per_station[station] = {var: [] for var in data_vars}
                data_per_station[station]['time'] = []

            for var in data_vars:
                data_per_station[station][var].append(ds[var].isel(stations=i).values)
            data_per_station[station]['time'].append(time)

        ds.close()

    for station in data_per_station:
        for key in data_per_station[station]:
            data_per_station[station][key] = np.concatenate(data_per_station[station][key])

    if save:
        print('All years loaded. Starting export...')
        os.makedirs(outdir, exist_ok=True)

        for station, data in data_per_station.items():
            outpath = os.path.join(outdir, f'{station}.npz')
            np.savez(os.path.join(outdir, 'global_coords.npz'), sigma=sigma, theta=theta)
            np.savez(outpath, **data)
            print(f'  ✓ Saved {station}.npz with {len(data["time"])} timesteps.')

        print(f'Extraction finished. Data saved to: {outdir}')
    else:
        print(f'Extraction complete. Data not saved (save=False).')

    return data_per_station, sigma, theta

# === Run station extraction ===
np.set_printoptions(threshold=np.inf)
ds, sigma, theta = process_station_data(
    model_base_path=base_path,
    outdir=postprocess_path,
    y0=year_start,
    y1=year_end,
    save=False
)

# === Build one big 4D array: (time, site, dir, freq) ===
site_names = list(ds.keys())
n_sites = len(site_names)
n_freq = len(sigma)
n_dir = len(theta)

# Get time length from first station
time_lens = [len(ds[site]["time"]) for site in site_names]
if not all(t == time_lens[0] for t in time_lens):
    raise ValueError("Mismatch in time lengths across stations, cannot stack directly.")

nt = time_lens[0]

# Load data
# Initialize big efth array
efth_all = np.zeros((nt, n_sites, n_dir, n_freq))
for i, site in enumerate(site_names):
    efth_all[:, i, :, :] = ds[site]["point_spectrum2d"]

# Apply scaling and remove last timestep if needed

efth_all = efth_all * 10 / (180 / np.pi)
efth_all = efth_all[:-1]
model_timesteps = ds[site_names[0]]["time"][:-1]  # use one station's time
site_coord = np.array(site_names)

# === Final DataArray ===
model_da = xr.DataArray(
    data=efth_all,
    dims=["time", "site", "dir", "freq"],
    coords=dict(
        time=model_timesteps,
        site=site_coord,
        dir=theta,
        freq=sigma,
    ),
    name="efth",
)
model_dset = model_da.to_dataset()

model_stats = model_dset.spec.stats(
    ["hs", "hmax", "tp", "tm01", "tm02", "tm_minus10", "dpm", "dm", "dspr", "swe"]
)

A12_model_stats = model_stats.isel(site=7)  # Select the 8th site A12
Hoorn_model_stats = model_stats.isel(site=2)  # Select the 3rd site Hoorn

def remove_hs_outliers(stats, threshold):
    """
    Remove outliers in 'hs' by setting values above the threshold to NaN.
    Returns a copy of the stats dataset with outliers removed.
    """
    stats_clean = stats.copy()
    stats_clean = stats_clean.drop_duplicates(dim='time')
    hs = stats_clean['hs']
    # Find times where hs > threshold
    outlier_times = hs['time'][hs > threshold]
    # Drop all entries at those times
    stats_clean = stats_clean.drop_sel(time=outlier_times)
    # Remove all entries at times where 'hs' is NaN
    nan_times = stats_clean['time'][np.isnan(stats_clean['hs'].values)]
    stats_clean = stats_clean.drop_sel(time=nan_times)
    return stats_clean

# Example usage:
A12_model_stats = remove_hs_outliers(A12_model_stats, threshold=50)
Hoorn_model_stats = remove_hs_outliers(Hoorn_model_stats, threshold=50)

def read_spectra(filepath,data_name):
    # Load .mat files
    Czz = scipy.io.loadmat(os.path.join(filepath, f'{data_name}_Czz.mat'))['Czz'].flatten()
    Th0 = scipy.io.loadmat(os.path.join(filepath, f'{data_name}_Th0.mat'))['Th0'].flatten()

    yds = datetime(2009, 2, 1, 12, 10, 0)  # A12

    vpm_Czz = 101
    vpm_Th0 = 96
    nCzz = len(Czz)
    nTh0 = len(Th0)
    nsp = nCzz // vpm_Czz

    # reconstruct time vector
    tt = np.array([yds + timedelta(minutes=10*i) for i in range(nsp)])

    # frequency and direction bins
    fw = np.concatenate(([0, 0.005], np.arange(0.015, 1.0, 0.01)))
    fd = np.concatenate(([0.03], np.arange(0.035, 0.5, 0.01)))
    nfd = len(fd)

    # time window
    ydi1, moi1, ddi1 = 2009, 2, 1
    ydi2, moi2, ddi2 = 2013, 12, 31
    ydr1 = datetime(ydi1, moi1, ddi1)
    ydr2 = datetime(ydi2, moi2, ddi2)

    ii = np.where((tt >= ydr1) & (tt <= ydr2))[0]
    ng = len(ii)
    jg = 0

    df = 0.01
    ff = np.arange(0.035, 0.485 + df, df)
    nf = len(ff)
    dtheta = 10 * np.pi / 180
    theta_r = np.arange(-np.pi, np.pi + dtheta, dtheta)
    ntheta = len(theta_r)

    Efdt = []
    Efdt_per_degree = []
    Hm0 = []
    thetap = []
    Tp = []

    for jsp in range(ii[0], ii[-1] + 1):
        jg += 1
        E = Czz[(jsp)*vpm_Czz:(jsp+1)*vpm_Czz]
        thetam = Th0[(jsp)*vpm_Th0:(jsp)*vpm_Th0+nfd]
        thetas = Th0[(jsp)*vpm_Th0+nfd:(jsp)*vpm_Th0+2*nfd]

        Ef = E[4:51]  # 5:51 in MATLAB is 4:51 in Python (0-based)
        tf = thetam[1:-1]
        sf = thetas[1:-1]

        s = np.maximum(np.round(2. / (sf * np.pi / 180) ** 2 - 1), 1)
        m = 2 * s

        Efd = np.zeros((ntheta, nf))
        for jf in range(nf):
            Dd = np.maximum(np.cos(0.5 * (theta_r - tf[jf] * np.pi / 180)) ** (m[jf]), np.finfo(float).eps)
            Ad = 1. / (np.sum(Dd) * dtheta) 
            Dd = Dd * Ad
            Efd[:, jf] = Ef[jf] * Dd

        Efd = Efd / 1e4  # to get to m2/Hz
        
        Efdt.append(Efd)
        Eff = np.sum(Efd, axis=0) * dtheta
        Ett = np.sum(Efd, axis=1) * df
        ig = np.where(ff > 0.04)[0]
        Hm0.append(4 * np.sqrt(np.sum(Eff[ig]) * df))
        it = np.argmax(Ett)
        is_ = np.argmax(Eff)
        thetap.append(theta_r[it] * 180 / np.pi)
        Tp.append(1. / ff[is_])

        # Convert to m2/degree/Hz
        dtheta_deg = dtheta * 180 / np.pi  # Convert radians to degrees (10°)
        Efd_per_degree = Efd.copy() / ( 180 / np.pi)  # Correct for external use
        Efdt_per_degree.append(Efd_per_degree)

    theta_r = theta_r * 180 / np.pi  # Convert to degrees
    theta_r = theta_r + 180 # Shift to positive degrees
    theta_r = (theta_r + 180) % 360 # Make sure the waves come from the right direction
    ttw = tt[ii]
    return {
        'ttw': ttw,
        'Hm0': np.array(Hm0),
        'thetap': np.array(thetap),
        'Tp': np.array(Tp),
        'theta_r': theta_r,
        'ff': ff,
        'Efdt': np.array(Efdt),
        'Efdt_per_degree': np.array(Efdt_per_degree)
    }
    
A12_spectra_data = read_spectra(filepath_data,'A12')
Hoorn_spectra_data = read_spectra(filepath_data,'H')

print(f"Units: data_Efdt is in m²/Hz, data_Efdt_per_degree is in m²/degree/Hz")


# # Create DataArray, with an added dimension (as in your original line)
A12_data_da = xr.DataArray(
    data=A12_spectra_data['Efdt_per_degree'],

    dims=["time","dir", "freq"],
    coords=dict(time=A12_spectra_data['ttw'],freq=A12_spectra_data['ff'], dir=A12_spectra_data['theta_r']),
    name="efth",
)

Hoorn_data_da = xr.DataArray(
    data=Hoorn_spectra_data['Efdt_per_degree'],

    dims=["time","dir", "freq"],
    coords=dict(time=Hoorn_spectra_data['ttw'],freq=Hoorn_spectra_data['ff'], dir=Hoorn_spectra_data['theta_r']),
    name="efth",
)

A12_data_dset = A12_data_da.to_dataset()
Hoorn_data_dset = Hoorn_data_da.to_dataset()

import numpy as np

def read_epl_spectra(filepath, ipart=1, xaver=1, egon=0, jy=3):
    """
    Reconstruct frequency-directional spectra for Euro Platform (EPL) observations.
    
    Parameters:
        filepath: folder where .mat files are stored
        ipart: 1 for first part, 2 for second part
        xaver: 1 if only for storm XAVER
        egon: 1 if only for storm EGON
        jy: year offset for the analysis (3 for XAVER)
        
    Returns:
        Dictionary with time series, wave parameters, and spectra
    """
    # Load appropriate part
    if ipart == 1:
        Czz = scipy.io.loadmat(os.path.join(filepath, 'H_Czz_EPL3_p1.mat'))['Czz10'].flatten()
        Th0 = scipy.io.loadmat(os.path.join(filepath, 'H_Th0_EPL3_p1.mat'))['Th010'].flatten()
        Sh0 = scipy.io.loadmat(os.path.join(filepath, 'H_Sh0_EPL3_p1.mat'))['Sh010'].flatten()
        yds = datetime(2010, 1, 1, 0, 0, 0)
    else:
        Czz = scipy.io.loadmat(os.path.join(filepath, 'H_Czz_EPL3_p2.mat'))['Czz10'].flatten()
        Th0 = scipy.io.loadmat(os.path.join(filepath, 'H_Th0_EPL3_p2.mat'))['Th010'].flatten()
        Sh0 = scipy.io.loadmat(os.path.join(filepath, 'H_Sh0_EPL3_p2.mat'))['Sh010'].flatten()
        yds = datetime(2016, 8, 5, 0, 0, 0)

    # Parameters
    vpm_Czz = 51
    vpm_Th0 = 51
    vpm_Sh0 = 51
    nCzz = len(Czz)
    nTh0 = len(Th0)
    nSh0 = len(Sh0)
    nsp = nCzz // vpm_Czz

    # Time vector (10-minute intervals)
    tt = np.array([yds + timedelta(minutes=10*i) for i in range(nsp)])

    # Frequency and directional bins
    fw = np.arange(0, 0.51, 0.01)
    fd = np.arange(0, 0.51, 0.01)
    nfd = len(fd)

    # Define start and end datetime
    ydi1, moi1, ddi1 = 2010 + jy, 1, 1
    if xaver == 1:
        moi1 = 12
    if egon == 1:
        moi1, ddi1 = 1, 11
    ydr1 = datetime(ydi1, moi1, ddi1, 0, 0, 0)

    ydi2 = 2010 + jy
    if jy == 6 and ipart == 1:
        moi2, ddi2 = 8, 5
        ydr2 = datetime(ydi2, moi2, ddi2, 23, 40, 0)
    else:
        moi2, ddi2 = 12, 31
        if xaver == 1:
            ddi2 = 10
        if egon == 1:
            moi2, ddi2 = 1, 15
        ydr2 = datetime(ydi2, moi2, ddi2, 0, 0, 0) + timedelta(days=1)

    # Hourly averaging
    Dhr = 1
    nhr = Dhr * 6  # number of 10-min intervals per hour
    ii = np.where((tt > ydr1) & (tt < ydr2))[0]
    ng = round(len(ii) / nhr)

    # Preallocate arrays
    Hm0 = []
    Tp = []
    Dspr_mean = []
    theta_mean = []
    fm_01 = []
    fm_10 = []
    fm_20 = []
    tth = []
    Efdt = []
    Efdt_per_degree = []

    df = 0.01
    ff = fw[4:51]  # MATLAB 5:51 -> Python 4:51
    nf = len(ff)
    dtheta = 10 * np.pi / 180
    theta_r = np.arange(-np.pi, np.pi + dtheta, dtheta)
    ntheta = len(theta_r)

    for jg, jsp in enumerate(range(ii[0], ii[-1], nhr)):
        E = np.zeros(vpm_Czz)
        thetamx = np.zeros(nfd)
        thetamy = np.zeros(nfd)
        thetas = np.zeros(nfd)

        for jhr in range(nhr):
            idx_Czz = slice((jsp + jhr) * vpm_Czz, (jsp + jhr + 1) * vpm_Czz)
            Htot = 4 * np.sqrt(np.sum(Czz[idx_Czz]) / 1e4 * 0.01)
            if Htot > 9:
                E[:] = np.nan
            else:
                E += Czz[idx_Czz]

            idx_Th0 = slice((jsp + jhr) * vpm_Th0, (jsp + jhr) * vpm_Th0 + nfd)
            thetamx += np.cos(np.deg2rad(Th0[idx_Th0]))
            thetamy += np.sin(np.deg2rad(Th0[idx_Th0]))

            idx_Sh0 = slice((jsp + jhr) * vpm_Sh0, (jsp + jhr) * vpm_Sh0 + nfd)
            thetas += Sh0[idx_Sh0]

        # Hourly mean
        E /= nhr
        thetamx /= nhr
        thetamy /= nhr
        thetas /= nhr
        thetam = np.rad2deg(np.arctan2(thetamy, thetamx))

        # Reconstruct directional spectrum
        Ef = E[4:51]
        tf = thetam[4:51]
        sf = thetas[4:51]

        s = np.maximum(np.round(2. / (sf * np.pi / 180)**2 - 1), 1)
        m = 2 * s

        Efd = np.zeros((ntheta, nf))
        for jf in range(nf):
            Dd = np.maximum(np.cos(0.5 * (theta_r - np.deg2rad(tf[jf]))) ** m[jf], np.finfo(float).eps)
            Ad = 1. / (np.sum(Dd) * dtheta)
            Dd *= Ad
            Efd[:, jf] = Ef[jf] * Dd

        Efd = Efd / 1e4  # m2/Hz
        Efdt.append(Efd)
        Efdt_per_degree.append(Efd / (180 / np.pi))  # per degree

        Eff = np.sum(Efd, axis=0) * dtheta
        Ett = np.sum(Efd, axis=1) * df
        ig = np.where(ff > 0.04)[0]
        Hm0.append(np.real(4 * np.sqrt(np.sum(Eff[ig]) * df)))
        it = np.argmax(Ett)
        is_ = np.argmax(Eff)
        theta_mean.append(tf.mean())
        Dspr_mean.append(sf.mean())
        Tp.append(1. / ff[is_])
        times_chunk = tt[jsp:jsp + nhr]
        timestamps = np.array([t.timestamp() for t in times_chunk])
        mean_time = datetime.fromtimestamp(np.mean(timestamps))
        # tth.append(mean_time)  # append once per hourly block
        # Instead of mean time:
        block_start = tt[jsp]
        # Or if you prefer the central hour mark:
        block_center = tt[jsp] - timedelta(minutes=10)

        tth.append(block_center)  # aligned to start of block

    # Convert theta_r to degrees, shift to 0-360
    theta_r = theta_r * 180 / np.pi  # Convert to degrees
    theta_r = theta_r + 180 # Shift to positive degrees
    theta_r = (theta_r + 180) % 360 # Make sure the waves come from the right direction

    ttw = tt[ii][::nhr]  # hourly timestamps

    return {
        'ttw': np.array(tth),
        'Hm0': np.array(Hm0),
        'thetap': np.array(theta_mean),
        'Tp': np.array(Tp),
        'theta_r': theta_r,
        'ff': ff,
        'Efdt': np.array(Efdt),
        'Efdt_per_degree': np.array(Efdt_per_degree)
    }

EPL_spectra_data_list = []
for jy in range(0, 6):
    EPL_spectra_data_list.append(read_epl_spectra(filepath_data, ipart=1, xaver=0, egon=0, jy=jy))
# Concatenate all years along the time dimension

# Concatenate each field across years
def concat_dicts(dicts, axis=0):
    out = {}
    for key in dicts[0]:
        if key in ["ff", "theta_r"]:  # static coords
            out[key] = dicts[0][key]
        elif isinstance(dicts[0][key], np.ndarray) and dicts[0][key].ndim > 0:
            out[key] = np.concatenate([d[key] for d in dicts], axis=axis)
        else:
            out[key] = np.array([d[key] for d in dicts])
    return out

EPL_spectra_data = concat_dicts(EPL_spectra_data_list)

EPL_data_da = xr.DataArray(
    data=EPL_spectra_data['Efdt_per_degree'],

    dims=["time","dir", "freq"],
    coords=dict(time=EPL_spectra_data['ttw'],freq=EPL_spectra_data['ff'], dir=EPL_spectra_data['theta_r']),
    name="efth",
)

EPL_data_dset = EPL_data_da.to_dataset()

A12_data_stats = A12_data_dset.spec.stats(
    ["hs", "hmax", "tp", "tm01", "tm02", "tm_minus10", "dpm", "dm", "dspr", "swe"]
)

Hoorn_data_stats = Hoorn_data_dset.spec.stats(
    ["hs", "hmax", "tp", "tm01", "tm02", "tm_minus10", "dpm", "dm", "dspr", "swe"]
)

EPL_data_stats = EPL_data_dset.spec.stats(
    ["hs", "hmax", "tp", "tm01", "tm02", "tm_minus10", "dpm", "dm", "dspr", "swe"]
)

A12_data_stats = remove_hs_outliers(A12_data_stats, threshold=50)
Hoorn_data_stats = remove_hs_outliers(Hoorn_data_stats, threshold=50)
EPL_data_stats = remove_hs_outliers(EPL_data_stats, threshold=50)

# Convert longitudes from -180..180 to 0..360
EPL_lon_old = 3.275037
EPL_lon = (3.275037 + 180) % 360
EPL_lat = 51.997799 # Euro platform
Hoorn_lon_old = 4.150286
Hoorn_lon = (4.150286 + 180) % 360
Hoorn_lat = 52.925353 # Platform Hoorn Q1-A
A12_lon_old = 3.817000
A12_lon = (3.817000 + 180) % 360
A12_lat = 55.417000 # A12 platform

path_ERA5 = '/gpfs/work3/0/ai4nbs/ERA5_data/data'
def extract_hourly_u10_v10(path_ERA5, years, lat, lon):
    """
    Extract hourly u10 and v10 at a specific lat/lon for given years.
    Returns:
        times: concatenated array of datetime64
        u10: concatenated array of u10 values
        v10: concatenated array of v10 values
    """
    u10_all = []
    v10_all = []
    time_all = []
    for year in years:
        print(f'Extracting data for year: {year}')
        u10_file = os.path.join(path_ERA5, '10m_u_component_of_wind', f'global_10m_u_component_of_wind_{year}.nc')
        v10_file = os.path.join(path_ERA5, '10m_v_component_of_wind', f'global_10m_v_component_of_wind_{year}.nc')
        ds_u10 = xr.open_dataset(u10_file)
        ds_v10 = xr.open_dataset(v10_file)
        # Find nearest grid point
        abs_lat = np.abs(ds_u10['latitude'].values - lat)
        abs_lon = np.abs(ds_u10['longitude'].values - lon)
        i_lat = abs_lat.argmin()
        i_lon = abs_lon.argmin()
        # Extract time series at nearest grid point
        u10 = ds_u10['u10'][:, i_lat, i_lon].values
        v10 = ds_v10['v10'][:, i_lat, i_lon].values
        times = ds_u10['valid_time'].values
        u10_all.append(u10)
        v10_all.append(v10)
        time_all.append(times)
        ds_u10.close()
        ds_v10.close()
    u10_all = np.concatenate(u10_all)
    v10_all = np.concatenate(v10_all)
    time_all = np.concatenate(time_all)
    # Convert all times to numpy.datetime64[ns]
    time_all = time_all.astype('datetime64[ns]')
    return time_all, u10_all, v10_all

# Example usage:
# times, u10, v10 = extract_hourly_u10_v10(path_ERA5, [2010,2011], EPL_lat, EPL_lon)

A12_time_uv, A12_u10, A12_v10 = extract_hourly_u10_v10(path_ERA5, [2009, 2010, 2011, 2012, 2013], A12_lat, A12_lon_old)
Hoorn_time_uv, Hoorn_u10, Hoorn_v10 = extract_hourly_u10_v10(path_ERA5, [2009, 2010, 2011, 2012, 2013], Hoorn_lat, Hoorn_lon_old)
EPL_time_uv, EPL_u10, EPL_v10 = extract_hourly_u10_v10(path_ERA5, [2010, 2011, 2012, 2013, 2014, 2015], EPL_lat, EPL_lon_old)

def match_common_times(da1, da2):
    """
    Finds the common times in both xarray DataArrays and returns both arrays
    with only the values at the common times.

    Parameters:
        da1, da2: xarray.DataArray
            DataArrays with a 'time' coordinate.

    Returns:
        da1_common, da2_common: xarray.DataArray
            DataArrays indexed only at the common times.
    """
    # Convert times to numpy.datetime64 for comparison
    # Make sure times are unique
    da1 = da1.copy()
    da2 = da2.copy()
    da1 = da1.drop_duplicates(dim='time')
    da2 = da2.drop_duplicates(dim='time')
    t1 = pd.to_datetime(da1['time'].values)
    t2 = pd.to_datetime(da2['time'].values)
    common_times = np.intersect1d(t1, t2)
    da1_common = da1.sel(time=common_times)
    da2_common = da2.sel(time=common_times)
    return da1_common, da2_common

def match_common_times_stats(model_da,data_da):
    common_model_da = model_da
    common_model_da, common_data_da = match_common_times(common_model_da, data_da)
    common_model_dset = common_model_da.to_dataset()
    common_data_dset = common_data_da.to_dataset()
    common_model_stats = common_model_dset.spec.stats(
        ["hs", "hmax", "tp", "tm01", "tm02", "tm_minus10", "dpm", "dm", "dspr", "swe"]
    )
    common_data_stats = common_data_dset.spec.stats(
        ["hs", "hmax", "tp", "tm01", "tm02", "tm_minus10", "dpm", "dm", "dspr", "swe"]
    )
    common_model_stats = remove_hs_outliers(common_model_stats, threshold=20)
    common_data_stats = remove_hs_outliers(common_data_stats, threshold=20)

    common_model_stats, common_data_stats = match_common_times(common_model_stats, common_data_stats)





    return common_model_da, common_data_da, common_model_stats, common_data_stats

A12_common_model_da, A12_common_data_da, A12_common_model_stats, A12_common_data_stats = match_common_times_stats(model_da[:, 7,:,:], A12_data_da)
Hoorn_common_model_da, Hoorn_common_data_da, Hoorn_common_model_stats, Hoorn_common_data_stats = match_common_times_stats(model_da[:, 2,:,:], Hoorn_data_da)
EPL_common_model_da, EPL_common_data_da, EPL_common_model_stats, EPL_common_data_stats = match_common_times_stats(model_da[:, 0,:,:], EPL_data_da)

# Remove outliers from the common model and data stats
# A12_common_model_stats = remove_hs_outliers(A12_common_model_stats, A12_common_data_stats, threshold=20)
# Hoorn_common_model_stats = remove_hs_outliers(Hoorn_common_model_stats, Hoorn_common_data_stats, threshold=20)
# A12_common_data_stats = remove_hs_outliers(A12_common_data_stats, A12_common_model_stats, threshold=20)
# Hoorn_common_data_stats = remove_hs_outliers(Hoorn_common_data_stats, Hoorn_common_model_stats, threshold=20)
# EPL_common_model_stats = remove_hs_outliers(EPL_common_model_stats, EPL_common_data_stats, threshold=20)
# EPL_common_data_stats = remove_hs_outliers(EPL_common_data_stats, EPL_common_model_stats, threshold=20)

A12_common_model_dset = A12_common_model_da.to_dataset()
Hoorn_common_model_dset = Hoorn_common_model_da.to_dataset()
A12_common_data_dset = A12_common_data_da.to_dataset()
Hoorn_common_data_dset = Hoorn_common_data_da.to_dataset()
EPL_common_model_dset = EPL_common_model_da.to_dataset()
EPL_common_data_dset = EPL_common_data_da.to_dataset()


def cut_depth_to_times(depth_data, size):
    return np.full(size, depth_data[0], dtype=depth_data.dtype)

A12_dep = cut_depth_to_times(A12_dep, len(A12_common_data_dset.coords['time']))
Hoorn_dep = cut_depth_to_times(Hoorn_dep, len(Hoorn_common_data_dset.coords['time']))
EPL_dep = cut_depth_to_times(EPL_dep, len(EPL_common_data_dset.coords['time']))

def rmse(obs, pred):
    """Root Mean Square Error with NaN handling"""
    if obs.size == 0:
        return np.nan
    return np.sqrt(np.mean((pred - obs) ** 2))

def bias(obs, pred):
    """Mean Bias (Mean Error) with NaN handling"""
    if obs.size == 0:
        return np.nan
    return np.mean(pred - obs)

def scatter_index(obs, pred):
    """Scatter Index: RMSE normalized by mean of observations, NaN safe"""
    if obs.size == 0:
        return np.nan
    return rmse(obs, pred) / np.mean(obs)

def angular_diffs(obs_angles, pred_angles):
    """Compute angular differences between observed and predicted angles."""
    obs_angles_rad = np.deg2rad(obs_angles)
    pred_angles_rad = np.deg2rad(pred_angles)
    angular_diffs_rad = np.arctan2(np.sin(pred_angles_rad - obs_angles_rad), np.cos(pred_angles_rad - obs_angles_rad))
    angular_diffs = np.rad2deg(angular_diffs_rad)
    return angular_diffs

def angular_RMSE(obs_angles, pred_angles):
    """Compute the angular RMSE between observed and predicted angles."""
    angular_diffs_values = angular_diffs(obs_angles, pred_angles)
    return np.sqrt(np.mean(angular_diffs_values ** 2)) if angular_diffs_values.size > 0 else np.nan

variable_list = ['hs', 'tp', 'tm01', 'tm02', 'tm_minus10']
angular_variable_list = ['dpm', 'dm', 'dspr']

var_metadata = {
    "hs": {"title": "Significant Wave Height", "unit": "m"},
    "tp": {"title": "Peak Period", "unit": "s"},
    "tm01": {"title": "Mean Wave Period (m01)", "unit": "s"},
    "tm02": {"title": "Mean Wave Period (m02)", "unit": "s"},
    "tm_minus10": {"title": "Mean Wave Period (m-10)", "unit": "s"},
    "dpm": {"title": "Peak Wave Direction", "unit": "°"},
    "dm": {"title": "Mean Wave Direction", "unit": "°"},
    "dspr": {"title": "Directional Spread", "unit": "°"},
    "wspd": {"title": "10m Wind Speed", "unit": "m/s"},
    "wdir": {"title": "10m Wind Direction", "unit": "°"},
    "dpt": {"title": "Water Depth", "unit": "m"},
    "dir": {"title": "Mean Wave Direction", "unit": "°"},
}

df_statistics = {}

start_time = np.datetime64('2013-12-01')
end_time = np.datetime64('2013-12-07T23:59:59')

for station_name, common_data_stats, common_model_stats in zip(
    ["A12", "EPL", "Hoorn"],
    [A12_common_data_stats.sel(time=slice(start_time, end_time)), EPL_common_data_stats.sel(time=slice(start_time, end_time)), Hoorn_common_data_stats.sel(time=slice(start_time, end_time))],
    [A12_common_model_stats.sel(time=slice(start_time, end_time)), EPL_common_model_stats.sel(time=slice(start_time, end_time)), Hoorn_common_model_stats.sel(time=slice(start_time, end_time))],
):
    df_statistics[station_name] = {}
    for variable in variable_list:
        obs = common_data_stats[variable].values
        pred = common_model_stats[variable].values

        print(f"Calculating statistics for {station_name} - {variable}")
        print(f"obs: {obs.shape}")
        print(f"pred: {pred.shape}")

        df_statistics[station_name][variable] = {
            "RMSE": rmse(obs, pred),
            "SI": scatter_index(obs, pred),
            "Bias": bias(obs, pred),
        }
    for angular_variable in angular_variable_list:
        obs = common_data_stats[angular_variable].values
        pred = common_model_stats[angular_variable].values
        df_statistics[station_name][angular_variable] = {
            "Angular RMSE": angular_RMSE(obs, pred),
        }

platforms = ["A12", "EPL", "Hoorn"]
variables = ["tp", "tm01", "tm_minus10", "tm02"]

for platform in platforms:
    print(f"Statistics for platform: {platform}")
    for variable in variables:
        stats = df_statistics[platform][variable]
        print(f"  {variable}:")
        print(f"    RMSE: {stats['RMSE']:.2f}")
        print(f"    Scatter Index: {stats['SI']:.2f}")
        print(f"    Bias: {stats['Bias']:.2f}")
    print()

outpath='/gpfs/work3/0/ai4nbs/hurry_wave/north_sea/05_postprocessing/Spectra'

print("Plotting spectra for specific timestamps...")
# Print datetime at index 42443 for each dataset (safe checks)
for name, dset in [("Hoorn", Hoorn_common_model_dset), ("A12", A12_common_model_dset)]:
    try:
        idx = 42443
        if idx < 0 or idx >= dset.dims['time']:
            print(f"{name}: index {idx} out of range (0..{dset.dims['time']-1})")
            continue
        tval = pd.to_datetime(dset['time'].isel(time=idx).values)
        print(f"model {name} time at index {idx}: {tval}")
    except Exception as e:
        print(f"{name}: error retrieving time at index {idx}: {e}")

print("Plotting spectra for specific timestamps...")
# Print datetime at index 42443 for each dataset (safe checks)
for name, dset in [("Hoorn", Hoorn_common_data_dset), ("A12", A12_common_data_dset)]:
    try:
        idx = 42443
        if idx < 0 or idx >= dset.dims['time']:
            print(f"{name}: index {idx} out of range (0..{dset.dims['time']-1})")
            continue
        tval = pd.to_datetime(dset['time'].isel(time=idx).values)
        print(f"data {name} time at index {idx}: {tval}")
    except Exception as e:
        print(f"{name}: error retrieving time at index {idx}: {e}")




# Print the datetime at index 34440 for EPL dataset
try:
    idx = 34440
    if idx < 0 or idx >= EPL_common_model_dset.dims['time']:
        print(f"EPL: index {idx} out of range (0..{EPL_common_model_dset.dims['time']-1})")
    else:
        tval = pd.to_datetime(EPL_common_model_dset['time'].isel(time=idx).values)
        print(f"model EPL time at index {idx}: {tval}")
except Exception as e:
    print(f"EPL: error retrieving time at index {idx}: {e}")


# Print the datetime at index 34440 for EPL dataset
try:
    idx = 34440
    if idx < 0 or idx >= EPL_common_data_dset.dims['time']:
        print(f"EPL: index {idx} out of range (0..{EPL_common_data_dset.dims['time']-1})")
    else:
        tval = pd.to_datetime(EPL_common_data_dset['time'].isel(time=idx).values)
        print(f"data EPL time at index {idx}: {tval}")
except Exception as e:
    print(f"EPL: error retrieving time at index {idx}: {e}")

# EPL model spectrum at 2013-12-06 00:00
ds_model_timestep = EPL_common_model_dset.isel(time=34440).spec
ds_model_timestep.spec.plot(cmap='viridis', 
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )

plt.title('HurryWave Model Spectrum at Euro Platform at 00:00 on 2013-12-06')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'EPL_Model_Spectrum_00' + '.png'), dpi=300)
plt.close()

# EPL data spectrum at 2013-12-06 00:00
ds_data_timestep = EPL_common_data_dset.isel(time=34440).spec
ds_data_timestep.spec.plot(cmap='viridis',
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )
plt.title('Euro Platform Observed Spectrum at 00:00 on 2013-12-06')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'EPL_Observed_Spectrum_00' + '.png'), dpi=300)
plt.close()



# Hoorn model spectrum at 2013-12-05 21:00
ds_model_timestep = Hoorn_common_model_dset.isel(time=42440).spec
ds_model_timestep.spec.plot(cmap='viridis',
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )
plt.title('HurryWave Model Spectrum at Hoorn Station at 21:00 on 2013-12-05')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'Hoorn_Model_Spectrum_21' + '.png'), dpi=300)
plt.close()


# Hoorn data spectrum at 2013-12-05 21:00

ds_data_timestep = Hoorn_common_data_dset.isel(time=42440).spec
ds_data_timestep.spec.plot(cmap='viridis',
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )
plt.title('Hoorn Station Observed Spectrum at 21:00 on 2013-12-05')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'Hoorn_Observed_Spectrum_21' + '.png'), dpi=300)
plt.close()

# Hoorn model spectrum at 2013-12-06 00:00
ds_model_timestep = Hoorn_common_model_dset.isel(time=42443).spec
ds_model_timestep.spec.plot(cmap='viridis',
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )
plt.title('HurryWave Model Spectrum at Hoorn Station at 00:00 on 2013-12-06')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'Hoorn_Model_Spectrum_00' + '.png'), dpi=300)
plt.close()

# Hoorn data spectrum at 2013-12-06 00:00
ds_data_timestep = Hoorn_common_data_dset.isel(time=42443).spec
ds_data_timestep.spec.plot(cmap='viridis',
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )
plt.title('Hoorn Station Observed Spectrum at 00:00 on 2013-12-06')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'Hoorn_Observed_Spectrum_00' + '.png'), dpi=300)
plt.close()

# A12 model spectrum at 2013-12-06 00:00
ds_model_timestep = A12_common_model_dset.isel(time=42443).spec
ds_model_timestep.spec.plot(cmap='viridis',
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )
plt.title('HurryWave Model Spectrum at A12 Platform at 00:00 on 2013-12-06')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'A12_Model_Spectrum_00' + '.png'), dpi=300)
plt.close()

# A12 data spectrum at 2013-12-06 00:00
ds_data_timestep = A12_common_data_dset.isel(time=42443).spec
ds_data_timestep.spec.plot(cmap='viridis',
                            cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
                            radii_ticks = [0.04, 0.1, 0.2, 0.3, 0.4],
                            radii_ticks_kwargs={'fontsize':12}
                            )
plt.title('A12 Platform Observed Spectrum at 00:00 on 2013-12-06')
plt.xticks(fontsize=12)
plt.yticks( fontsize=12)
plt.savefig(os.path.join(outpath, 'A12_Observed_Spectrum_00' + '.png'), dpi=300)
plt.close()

