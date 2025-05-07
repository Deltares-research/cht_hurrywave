# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:26:26 2022
run in cartop_env2
@author: keesn
@author: kvanasselt
"""

# Modules
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd
import os
from datetime import datetime
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import math

# cartopy
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
import cartopy.io.img_tiles as cimgt
import cartopy.mpl.geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Make list of pickles from Jake
filenames        = []
filenames.append(r"c:\projects\fhics\drifters_Helene\hurricane_helene_drifter_data_v0.pickle")

# Make lists of netcdfs
model_netcdfs   = []

main_dir_models = r"p:\11206085-onr-fhics\03_cosmos\scenarios\hurricane_helene"

# Extract all folder names in storm_data_base that start with "coamps_tc"
coamps_tc_folders_name = [folder for folder in os.listdir(main_dir_models) if folder.startswith("hurricane_helene_tau_2_dt_1_dx_0_1")]
scenario = "20240925_00z"

for folder in coamps_tc_folders_name:
    model_netcdfs.append(r"p:\11206085-onr-fhics\03_cosmos\scenarios\hurricane_helene\{}\{}\models\hurrywave_gom\input".format(folder, scenario))
    
grid_sizes      = 0.1
hurricane_name = "Helene"

# Destout
destout     = r'c:\projects\fhics\drifters_Helene\validate_HW_2'

if not os.path.exists(destout):
    os.mkdir(destout)
os.chdir(destout)

# Private functions
def MEM_directionalestimator(a1, a2, b1, b2, en, convert):
    """
    Calculate the Maximum Entropy Method estimate of the Directional Distribution of a wave field.
    """
    # Switch to Krogstad notation
    d1 = np.array(a1).reshape(-1)
    d2 = np.array(b1).reshape(-1)
    d3 = np.array(a2).reshape(-1)
    d4 = np.array(b2).reshape(-1)
    en = np.array(en).reshape(-1)
    c1 = d1 + 1j * d2
    c2 = d3 + 1j * d4
    p1 = (c1 - c2 * np.conj(c1)) / (1 - np.abs(c1) ** 2)
    p2 = c2 - c1 * p1
    x1 = 1 - p1 * np.conj(c1) - p2 * np.conj(c2)
    
    # Define directional domain
    dtheta  = 2
    direc   = np.arange(0, 360, dtheta)
    
    # Get distribution with "dtheta" degree resolution
    dr = np.pi / 180
    S = np.zeros((len(en), len(direc)), dtype=np.complex128)
    for n, alpha in enumerate(direc):
        e1 = np.cos(alpha * dr) - 1j * np.sin(alpha * dr)
        e2 = np.cos(2 * alpha * dr) - 1j * np.sin(2 * alpha * dr)
        y1 = np.abs(1 - p1 * e1 - p2 * e2) ** 2
        S[:, n] = x1 / y1
    S = S.real
    
    # Normalize each frequency band
    tot = np.sum(S, axis=1) * dtheta
    Sn = np.divide(S, tot.reshape(-1, 1))
    
    # Calculate energy density
    E = Sn * en.reshape(-1, 1)
    
    if convert == 0:
        NE = E
        NS = Sn
    elif convert == 1:
        ndirec = np.abs(direc - 360)
        ndirec = (ndirec + 180) % 360
        NE = np.zeros_like(E)
        NS = np.zeros_like(Sn)
        for ii, d in enumerate(direc):
            ia = np.where(ndirec == d)
            if ia[0].size != 0:
                NE[:, ii] = E[:, ia[0][0]]
                NS[:, ii] = Sn[:, ia[0][0]]
            else:
                print("\n !!! Error converting to geographic coordinate frame !!!")
    else:
        NE, NS = None, None
    return NS, NE

    # Convert to nautical convention (clockwise from north)
    angle_nautical_degrees = 90.0 - angle_cartesian_degrees
    
    # Ensure the result is within the range [0, 360)
    angle_nautical_degrees %= 360.0
    
    return angle_nautical_degrees
import wave_stats

# Define a box filter function
def box_filter_1D(data, window_size):
    result = np.copy(data)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2)
        valid_data  = data[start:end+1][~np.isnan(data[start:end+1])]
        result[i]   = np.nanmean(valid_data) if valid_data.size else np.nan
    return result

# Define a 2D box filter
def box_filter_2d(data, window_size_x, window_size_y):
    result = np.copy(data)
    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            start_x     = max(0, i - window_size_x // 2)
            end_x       = min(rows, i + window_size_x // 2)
            start_y     = max(0, j - window_size_y // 2)
            end_y       = min(cols, j + window_size_y // 2)
            sub_matrix  = data[start_x:end_x+1, start_y:end_y+1]
            valid_data  = sub_matrix[~np.isnan(sub_matrix)]
            result[i, j] = np.nanmean(valid_data) if valid_data.size else np.nan
    return result


# Get best track
from shapely.geometry import LineString
track       = gpd.read_file(r'c:\projects\fhics\drifters_{}\track.geojson'.format(hurricane_name))
# lineStringObj = LineString( [[a.x, a.y] for a in track.geometry.values] )


for model_netcdf, model_name in zip(model_netcdfs[:1], coamps_tc_folders_name[:1]):

    destout_model = os.path.join(destout, model_name)

    if not os.path.exists(destout_model):
        os.mkdir(destout_model)

    # load Hurrywave results
    nc_name             = os.path.join(model_netcdf, 'hurrywave_his.nc')
    ds                  = xr.open_dataset(nc_name)
    x                   = ds.station_x
    y                   = ds.station_y
    names               = ds.station_name
    hm0                 = ds.point_hm0
    wavdir              = ds.point_wavdir
    dirspr              = ds.point_dirspr
    tp                  = ds.point_tp/1.22                      # tm
    times               = ds.time.values
    times               = times.astype(np.float64)/1000         # ms to s
    ds.close()

    # Load Hurrywave spectra
    nc_name             = os.path.join(model_netcdf, 'hurrywave_sp2.nc')
    ds                  = xr.open_dataset(nc_name)
    point_spectrum2d    = ds.point_spectrum2d  
    theta               = ds.theta
    sigma               = ds.sigma
    ds.close()

    # Define empty dict
    obs_results_all     = []

    # Loop over file
    parameters_wanted = ['wave_height', 'wave_period', 'directional_spreading', 'wave_direction']
    for filename in filenames:

        # Read the Pandas dataframe 
        print('Started with ' + filename)
        df          = pd.read_pickle(filename)
        id          = list(df.keys())

        # Loop over the dictonary 
        for index, (key, df_now) in enumerate(df.items()):

            # Specific type of data (Spotter, microswift, etc.)
            print(' working on ' + id[index])
            df_now      = df[key]
            ids         = list(df_now.keys())

            if index not in [0, 2]:
                continue

            # Going over each type of drifter
            for index2, (key2, df_now2) in enumerate(df_now.items()):

                # Get new one
                print(' => ' + ids[index2])
                df_now2                                 = df_now[key2]

                # Make new dict
                obs_results                             = {}
                obs_results['name']                     = ids[index2]
                obs_results['time']                     = df_now2.index.values
                obs_results['x']                        = df_now2.longitude.values
                obs_results['y']                        = df_now2.latitude.values
                obs_results['height_observed']          = np.copy(df_now2.significant_height.values) 
                obs_results['height_modeled']           = np.copy(df_now2.significant_height.values)

                obs_results['period_observed']          = np.copy(df_now2.mean_period.values) 
                obs_results['period_modeled']           = np.copy(df_now2.mean_period.values)

                obs_results['direction_observed']       = np.copy(df_now2.mean_direction.values) 
                obs_results['direction_modeled']        = np.copy(df_now2.mean_direction.values)

                obs_results['spreading_observed']       = np.copy(df_now2.mean_directional_spread.values) 
                obs_results['spreading_modeled']        = np.copy(df_now2.mean_directional_spread.values)

                obs_results['frequency']                = np.copy(df_now2.frequency.values) 
                obs_results['energy_observed']          = np.copy(df_now2.energy_density.values) 
                obs_results['energy_modeled']           = np.copy(df_now2.energy_density.values)

                # Create new destout
                destout_spectra = os.path.join(destout_model, obs_results['name'])
                if not os.path.exists(destout_spectra):
                    os.mkdir(destout_spectra)

                # Apply time filter on wave variables
                window_hours                            = (df_now2.index[2] - df_now2.index[1]).total_seconds() / 3600
                window_hours                            = np.round(window_hours*6)/6
                window_size                             = int(1/window_hours)

                # Find estimate of sigma value
                nn_start                                = 0
                sigma_values                            = df_now2.frequency.values[nn_start]
                while isinstance(sigma_values, float):
                    nn_start      = nn_start+1
                    sigma_values  = df_now2.frequency.values[nn_start]
                dsigma                                  = sigma_values[1] - sigma_values[0]
                sigma_size                              = int(0.02 / dsigma)

                # Simply per buoy type
                if index == 0:  sigma_size = 2
                if index == 1:  sigma_size = 5
                if index == 2:  sigma_size = 2

                E_values                                = df_now2.energy_density.values
                E_values                                = np.array([a if isinstance(a, np.ndarray) else [np.nan]*39 for a in E_values])
                E_smooth                                = box_filter_2d(E_values, window_size, sigma_size)

                a1_values                               = df_now2.a1.values
                a1_values                               = np.array([a if isinstance(a, np.ndarray) else [np.nan]*39 for a in a1_values])
                a1_smooth                               = box_filter_2d(a1_values, window_size, sigma_size)
                a2_values                               = df_now2.a2.values
                a2_values                               = np.array([a if isinstance(a, np.ndarray) else [np.nan]*39 for a in a2_values])
                a2_smooth                               = box_filter_2d(a2_values, window_size, sigma_size)

                b1_values                               = df_now2.b1.values
                b1_values                               = np.array([a if isinstance(a, np.ndarray) else [np.nan]*39 for a in b1_values])
                b1_smooth                               = box_filter_2d(b1_values, window_size, sigma_size)
                b2_values                               = df_now2.b2.values
                b2_values                               = np.array([a if isinstance(a, np.ndarray) else [np.nan]*39 for a in b2_values])
                b2_smooth                               = box_filter_2d(b2_values, window_size, sigma_size)

                # Loop over index (or time component)
                for index, row in enumerate(df_now2.values):
                    
                    # NaN modeled
                    obs_results['height_modeled'][index]        = np.nan
                    obs_results['period_modeled'][index]        = np.nan
                    obs_results['direction_modeled'][index]     = np.nan
                    obs_results['spreading_modeled'][index]     = np.nan

                    # Is this point in space we want?
                    if df_now2.latitude[index] > 15  and df_now2.latitude[index] < 80 and df_now2.longitude[index] > -110 and df_now2.longitude[index] < 0 and np.datetime64(df_now2.index[index]) <= datetime(2024,9,27,12,0,0) and np.datetime64(df_now2.index[index]) >= datetime(2024,9,25):
                            # Does this one exist?
                            ds.close()
                            timestamp           = np.datetime64(df_now2.index[index])
                            timestamp           = timestamp.astype(np.float64)
                            distances_points    = np.sqrt((df_now2.longitude[index] - x)**2 + (df_now2.latitude[index] - y)**2)
                            idwanted            = np.where(distances_points < grid_sizes*np.sqrt(2))
                            idwanted            = idwanted[0]
                            
                            # Make better choice: using the nearest value
                            if len(idwanted) > 1:
                                possible_distances  = distances_points[idwanted]
                                idwanted2           = np.where(possible_distances == np.min(possible_distances))
                                idwanted            = idwanted[idwanted2]
                            
                            # Get parameters
                            for option, parameter in enumerate(parameters_wanted):

                                # Change main options
                                if option == 0: name = 'significant_height';         name2 = 'hm0'
                                if option == 1: name = 'mean_period';                name2 = 'tp'
                                if option == 2: name = 'mean_direction';             name2 = 'wavdir'
                                if option == 3: name = 'mean_directional_spread';    name2 = 'dirspr'

                                # Find values
                                if idwanted.size == 0:
                                    model                   = np.NaN
                                    obs                     = df_now2[name][index]
                                else:
                                    
                                    # Get correct values
                                    if option == 0: wanted                  = np.squeeze(hm0[:,idwanted[0]].values)
                                    if option == 1: wanted                  = np.squeeze(tp[:,idwanted[0]].values)
                                    if option == 2: wanted                  = np.squeeze(wavdir[:,idwanted[0]].values)
                                    if option == 3: wanted                  = np.squeeze(dirspr[:,idwanted[0]].values)
                                    
                                    # Interpolate
                                    func                    = interp1d(times, wanted)
                                    if timestamp < np.min(times) or timestamp > np.max(times):
                                        model                   = np.NaN
                                        obs                     = df_now2[name][index]
                                    else:
                                        model                   = func(timestamp)
                                        obs                     = df_now2[name][index]
                                    
                                    # Also save a wave height at the 'nearest time stamp
                                    if option == 0:
                                        timewanted          = np.abs(times - timestamp)
                                        timewanted          = np.where(timewanted == np.min(timewanted))
                                        if np.squeeze(timewanted[0].shape) > 1:
                                            timewanted      = timewanted[0][0]
                                            model_Hs        = wanted[timewanted]
                                        else:
                                            timewanted      = timewanted[0]
                                            model_Hs        = wanted[timewanted]

                                # Store data
                                if option == 0: 
                                    obs_results['height_observed'][index]       = obs
                                    obs_results['height_modeled'][index]        = model
                                if option == 1: 
                                    obs_results['period_observed'][index]       = obs
                                    obs_results['period_modeled'][index]        = model
                                if option == 2: 
                                    obs_results['direction_observed'][index]    = obs
                                    obs_results['direction_modeled'][index]     = model
                                if option == 3: 
                                    obs_results['spreading_observed'][index]    = obs
                                    obs_results['spreading_modeled'][index]     = model

                                # Get wave spectra
                                if option == 3:
                                    if (np.isnan(model) == False) and (np.isnan(obs) == False):
                                
                                        # Only for large waves
                                        if obs_results['height_modeled'][index] > 0.00:

                                            # single point_spectrum2d(time,stations,theta,sigma)
                                            timewanted          = np.abs(times - timestamp)
                                            timewanted          = np.where(timewanted == np.min(timewanted))
                                            if np.squeeze(timewanted[0].shape) > 1:
                                                timewanted      = timewanted[0][0]
                                            else:
                                                timewanted = timewanted[0]
                                            spectra_model       = np.squeeze(point_spectrum2d[timewanted,idwanted[0], :,:])
                                            spectra_model       = spectra_model.values
                                            spectra_model       = np.transpose(spectra_model)
                                            spectra_model_1D    = np.trapz(spectra_model)
                                            Hm0_est             = 4 * np.sqrt(np.trapz(spectra_model_1D, sigma.values))
                                            func                = interp1d(sigma.values, spectra_model_1D)

                                            # Store data
                                            obs_results['energy_observed'][index]   = df_now2.energy_density[index]
                                            obs_results['energy_modeled'][index]    = spectra_model_1D

                                            # Store frequency too
                                            frequency_observed                      = df_now2.frequency[index]
                                            frequency_modeled                       = sigma.values

                                            # Estimate 2D spectra from buoy
                                            a1                                      = df_now2.a1[index]
                                            a2                                      = df_now2.a2[index]
                                            b1                                      = df_now2.b1[index]
                                            b2                                      = df_now2.b2[index]
                                            spectra_observed                        = MEM_directionalestimator(a1, a2, b1, b2, df_now2.energy_density[index], 0)
                                            spectra_observed                        = spectra_observed[1]*2
                                            
                                            spectra_observed_smooth                 = MEM_directionalestimator(a1_smooth[index], a2_smooth[index], b1_smooth[index], b2_smooth[index], E_smooth[index], 0)
                                            spectra_observed_smooth                 = spectra_observed_smooth[1]*2

                                            # I dont know why the MEM needs a factor 2 here
                                            # I also dont know why I need to set the directional_bin_width_deg to 1

                                            # Check 1D spectra 
                                            plt.plot(frequency_observed,(np.trapz(spectra_observed)))
                                            plt.plot(frequency_observed,(np.trapz(spectra_observed)))
                                            plt.plot(frequency_observed,df_now2.energy_density[index], '--')
                                            plt.plot(frequency_modeled, spectra_model_1D)

                                            # # Back calculate a1,a2, b1, b2 (from observations)
                                            # spectra_observed[np.isnan(spectra_observed)] = 0
                                            # spectra_observed_smooth[np.isnan(spectra_observed_smooth)] = 0
                                            # directional_bin_width_deg   = 1
                                            # dirs_r                      = np.linspace(0, 2*np.pi, 180)
                                            # S1, a1s, a2s, b1s, b2s      = wave_stats.to_Fourier( spectra_observed_smooth, frequency_observed, dirs_r, directional_bin_width_deg, faxis=0, daxis=1 )
                                            # S1b, a1s, a2s, b1s, b2s     = wave_stats.to_Fourier( spectra_observed, frequency_observed, dirs_r, directional_bin_width_deg, faxis=0, daxis=1 )

                                            # # Calculate a1,a2,b1,b2 from HurryWave
                                            # directional_bin_width_deg   = np.round(ds.theta.values[2] - ds.theta.values[1])
                                            # directional_bin_width_deg   = 1
                                            # dirs_r                      = ds.theta.values * (np.pi / 180)
                                            # S2a, a1s, a2s, b1s, b2s     = wave_stats.to_Fourier( spectra_model, frequency_modeled, dirs_r, directional_bin_width_deg, faxis=0, daxis=1 )
                                            # dirs_r                      = np.linspace(0, 2*np.pi, 36)
                                            # S2b, a1s, a2s, b1s, b2s     = wave_stats.to_Fourier( spectra_model, frequency_modeled, dirs_r, directional_bin_width_deg, faxis=0, daxis=1 )

                                            # # Compute wave heights (check)
                                            # Hm0_observed0   = obs_results['height_observed'][index]
                                            # Hm0_observed1   = wave_stats.calc_Hs_1d( S1, frequency_observed )       # smooth; therefore different
                                            # Hm0_observed1b  = wave_stats.calc_Hs_1d( S1b, frequency_observed )
                                            # Hm0_observed2   = wave_stats.calc_Hs_1d( df_now2.energy_density[index], frequency_observed )
                                            # Hm0_modeled0    = obs_results['height_modeled'][index]
                                            # Hm0_modeled1a   = wave_stats.calc_Hs_1d( S2a, frequency_modeled )
                                            # Hm0_modeled1b   = wave_stats.calc_Hs_1d( S2b, frequency_modeled )
                                            # Hm0_modeled2    = wave_stats.calc_Hs_1d( spectra_model_1D, frequency_modeled )

                                            # # Compute wave period
                                            # Tm_observed0    = obs_results['period_observed'][index]
                                            # Tm_observed1    = wave_stats.calc_TM01_1d( S1, frequency_observed )       # smooth; therefore different
                                            # Tm_observed2    = wave_stats.calc_TM01_1d( df_now2.energy_density[index], frequency_observed )
                                            # Tm_modeled0     = obs_results['period_modeled'][index]
                                            # Tm_modeled1     = wave_stats.calc_TM01_1d( S2a, frequency_modeled )
                                            # Tm_modeled2     = wave_stats.calc_TM01_1d( S2b, frequency_modeled )
                                            # Tm_modeled3     = wave_stats.calc_TM01_1d( spectra_model_1D, frequency_modeled )

                                            # # Overwrite period
                                            # obs_results['period_observed'][index]   = Tm_observed1
                                            # obs_results['period_modeled'][index]    = Tm_modeled1
                                            # obs_results['height_observed'][index]   = Hm0_observed1
                                            # obs_results['height_modeled'][index]    = Hm0_modeled1a

                                            # Plot energy and whatnot
                                            # Create a figure with five subplots (1 row, 5 columns)
                                            # fig, axs = plt.subplots(1, 5, figsize=(15, 5))

                                            # # Names
                                            # htext1              = f'HurryWave {obs_results["height_modeled"][index]:.1f} m & {obs_results["period_modeled"][index]:.1f}' +'s' 
                                            # htext2              = f'Observed {obs_results["height_observed"][index]:.1f} m & {obs_results["period_observed"][index]:.1f}' +'s' 
                                            # #axs[0].text(x_pos, y_pos, htext1, horizontalalignment='right', verticalalignment='top')
                                            # #axs[0].text(x_pos, y_pos - 0.05, htext2, horizontalalignment='right', verticalalignment='top')

                                            # # Subplot 1: Energy Density
                                            # axs[0].plot(frequency_observed, E_values[index], label="observed instant", color='gray',alpha=0.2)
                                            # for it in range(-window_size, window_size+1):
                                            #     if ((index+it) > 0) and ((index+it)<len(E_smooth)):
                                            #         E_plotting = E_values[index+it]
                                            #         if not np.isnan(E_plotting[0]):
                                            #             axs[0].plot(frequency_observed, E_plotting, color='gray',alpha=0.1)
                                            # axs[0].plot(frequency_observed, E_smooth[index], label=htext2, color='black')
                                            # axs[0].plot(frequency_modeled, spectra_model_1D, label=htext1)
                                            # axs[0].set_xlabel("frequency [Hz]")
                                            # axs[0].set_ylabel("energy density [m$^2$/Hz]")
                                            # axs[0].set_xlim(0, 0.5)  # Corrected line to set x-axis limits
                                            # axs[0].set_yscale('log')
                                            # axs[0].legend()

                                            # # Subplot 2: a1
                                            # axs[1].plot(frequency_observed, a1_values[index], label="observed_instant", color='gray',alpha=0.2)
                                            # for it in range(-window_size, window_size+1):
                                            #     if ((index+it) > 0) and ((index+it)<len(E_smooth)):
                                            #         E_plotting = a1_values[index+it]
                                            #         if not np.isnan(E_plotting[0]):
                                            #             axs[1].plot(frequency_observed, E_plotting, color='gray',alpha=0.1)                           
                                            # axs[1].plot(frequency_observed, a1_smooth[index], label="observed_smooth", color='black')
                                            # axs[1].plot(frequency_modeled, a1s, label="HurryWave")
                                            # axs[1].set_xlabel("frequency [Hz]")
                                            # axs[1].set_ylabel("a1")
                                            # axs[1].set_title("a1")
                                            # axs[1].set_xlim(0, 0.5)  # Corrected line to set x-axis limits
                                            # axs[1].set_ylim(-1, +1)  

                                            # # Subplot 3: a2
                                            # axs[2].plot(frequency_observed, a2_values[index], label="observed_instant", color='gray',alpha=0.2)
                                            # for it in range(-window_size, window_size+1):
                                            #     if ((index+it) > 0) and ((index+it)<len(E_smooth)):
                                            #         E_plotting = a2_values[index+it]
                                            #         if not np.isnan(E_plotting[0]):
                                            #             axs[2].plot(frequency_observed, E_plotting, color='gray',alpha=0.1)                                    
                                            # axs[2].plot(frequency_observed, a2_smooth[index], label="observed_smooth", color='black')
                                            # axs[2].plot(frequency_modeled, a2s, label="HurryWave")
                                            # axs[2].set_xlabel("frequency [Hz]")
                                            # axs[2].set_ylabel("a2")
                                            # axs[2].set_title("a2")
                                            # axs[2].set_xlim(0, 0.5)  # Corrected line to set x-axis limits
                                            # axs[2].set_ylim(-1, +1)  

                                            # # Subplot 4: b1
                                            # axs[3].plot(frequency_observed, b1_values[index], label="observed_instant", color='gray',alpha=0.2)
                                            # for it in range(-window_size, window_size+1):
                                            #     if ((index+it) > 0) and ((index+it)<len(E_smooth)):
                                            #         E_plotting = b1_values[index+it]
                                            #         if not np.isnan(E_plotting[0]):
                                            #             axs[3].plot(frequency_observed, E_plotting, color='gray',alpha=0.1)                                        
                                            # axs[3].plot(frequency_observed, b1_smooth[index], label="observed_smooth", color='black')
                                            # axs[3].plot(frequency_modeled, b1s, label="HurryWave")
                                            # axs[3].set_xlabel("frequency [Hz]")
                                            # axs[3].set_ylabel("b1")
                                            # axs[3].set_title("b1")
                                            # axs[3].set_xlim(0, 0.5)  # Corrected line to set x-axis limits
                                            # axs[3].set_ylim(-1, +1)  

                                            # # Subplot 5: b2
                                            # axs[4].plot(frequency_observed, b2_values[index], label="observed_instant", color='gray',alpha=0.2)
                                            # for it in range(-window_size, window_size+1):
                                            #     if ((index+it) > 0) and ((index+it)<len(E_smooth)):
                                            #         E_plotting = b2_values[index+it]
                                            #         if not np.isnan(E_plotting[0]):
                                            #             axs[4].plot(frequency_observed, E_plotting, color='gray',alpha=0.1)                                        
                                            # axs[4].plot(frequency_observed, b2_smooth[index], label="observed_smooth", color='black')
                                            # axs[4].plot(frequency_modeled, b2s, label="HurryWave")
                                            # axs[4].set_xlabel("frequency [Hz]")
                                            # axs[4].set_ylabel("b2")
                                            # axs[4].set_title("b2")
                                            # axs[4].set_xlim(0, 0.5)  # Corrected line to set x-axis limits
                                            # axs[4].set_ylim(-1, +1)  

                                            # # Adjust the layout
                                            # plt.tight_layout()

                                            # # Show the plots
                                            # dt                  = df_now2.index[index]
                                            # filename_format     = '%Y%m%d_%H%M%S'  # YearMonthDay_HourMinuteSecond
                                            # date_string         = dt.strftime(filename_format)
                                            # axs[0].set_title(obs_results['name'] + ' ' + date_string)
                                            # figure_png          = 'moments_' + date_string + '.png'
                                            # figure_png          = os.path.join(destout_spectra, figure_png)
                                            # plt.savefig(figure_png, format='png', dpi=150)
                                            # plt.close()

                                            # # Define the color map ranges and colors (for spectra)
                                            # bounds  = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
                                            # norm    = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

                                            # # Plot this
                                            # id_low                              = spectra_model < bounds[0]
                                            # spectra_model[id_low]               = np.nan
                                            # id_low                              = spectra_observed_smooth < bounds[0]
                                            # spectra_observed_smooth[id_low]     = np.nan

                                            # # Make figure
                                            # plt.close()
                                            # fig, ax                     = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, constrained_layout=True)
                                            # fig.set_size_inches(8.3, 11.7/2.5)
                                            # reversed_cmap               = plt.cm.get_cmap('Spectral_r')
                                            # [nx,nt]                     = np.shape(spectra_observed_smooth)
                                            # radians                     = np.linspace(0, 2 * np.pi, nt)
                                            # T, R                        = np.meshgrid(radians, frequency_observed)
                                            # c1                          = ax[0].pcolor(T, R, spectra_observed_smooth,norm=norm, cmap=reversed_cmap)
                                            # ax[0].grid(True)
                                            # ax[0].set_rmax(0.5)  # Set the maximum radial value
                                            # ax[0].set_title('observed at ' + obs_results['name'])
                                            # ax[0].text(-0.1,-0.2, 'created with Maximum Entropy Method (MEM) \n with box filter of 1 hours and 0.02 Hz', transform=ax[0].transAxes)

                                            # # Modeled
                                            # [nx,nt]             = np.shape(spectra_model)
                                            # radians             = np.linspace(0, 2 * np.pi, nt)
                                            # T, R                = np.meshgrid(radians, sigma.values)
                                            # c2                  = ax[1].pcolor(T, R, spectra_model,norm=norm, cmap=reversed_cmap)
                                            # ax[1].grid(True)
                                            # ax[1].set_rmax(0.5)  # Set the maximum radial value
                                            # text_wanted = r'modeled with HurryWave ($\Delta$$\sigma$ 12 - $\Delta$$\theta$ 36)'
                                            # ax[1].text(-0.1,-0.2, text_wanted, transform=ax[1].transAxes)

                                            # # Create a single colorbar for both subplots
                                            # cbar = fig.colorbar(c2, ax=[ax[0], ax[1]], extend="max")
                                            # cbar.set_label('energy density [m$^2$/Hz]')

                                            # # Get some information on this time stamp
                                            # htext1              = f'H_s - {obs_results["height_observed"][index]:.1f} - {obs_results["height_modeled"][index]:.1f}' +'\meter' 
                                            # htext2              = f'T_m - {obs_results["period_observed"][index]:.1f} - {obs_results["period_modeled"][index]:.1f}' +'\seconds' 
                                            # htext3              = f'D_m - {obs_results["direction_observed"][index]:.0f} - {obs_results["direction_modeled"][index]:.0f}' +'\degrees' 

                                            # # Print
                                            # dt                  = df_now2.index[index]
                                            # filename_format     = '%Y%m%d_%H%M%S'  # YearMonthDay_HourMinuteSecond
                                            # date_string         = dt.strftime(filename_format)
                                            # ax[1].set_title('model at ' + date_string)
                                            # figure_png          = 'spectrum2D_' + date_string + '.png'
                                            # figure_png          = os.path.join(destout_spectra, figure_png)
                                            # plt.savefig(figure_png, format='png', dpi=150)
                                            # plt.close()


                                
                    else:

                        # print('point not found - ' + str(df_now.latitude[index]) + ' & ' + str(df_now.longitude[index]) + ' time ' + str(df_now.index[index]) )
                        point_not_found = 1

                # Done with this drifter: make plots per parameters
                for option, parameter in enumerate(parameters_wanted):

                    # Flavors
                    if option == 0: name0 = 'height_modeled';       name1 = 'height_observed';      name2 = 'significant wave height ($H_s$) [m]'; name3 = '_Hs'
                    if option == 1: name0 = 'period_modeled';       name1 = 'period_observed';      name2 = 'mean wave period ($T_m0$) [s]'; name3 = '_Tm'
                    if option == 2: name0 = 'direction_modeled';    name1 = 'direction_observed';   name2 = 'mean wave direction ($D_m$) [$\circ$]'; name3 = '_Dm'
                    if option == 3: name0 = 'spreading_modeled';    name1 = 'spreading_observed';   name2 = 'mean wave spreading ($s$) [$\circ$]'; name3 = '_s'

                    # Compute skill- Hurrywave
                    actual          = np.squeeze(obs_results[name0])          # observations
                    pred            = np.squeeze(obs_results[name1])          # model
                    actual, pred    = np.array(actual), np.array(pred)
                    idwanted1       = np.isfinite(actual)
                    idwanted2       = np.isfinite(pred) 
                    idwanted        = idwanted1 & idwanted2
                    actual          = actual[idwanted]
                    pred            = pred[idwanted]
                    bias            = np.nanmean(actual - pred)
                    MAE             = np.nanmean(abs(actual - pred))
                    RMSE            = np.sqrt(np.square(np.subtract(actual,pred)).mean())
                    SCI             = RMSE/np.nanmean(pred)

                    # Make header
                    fig, ax = plt.subplots() # for landscape
                    fig.set_size_inches(8.3, 11.7/2)

                    # Subplot 1: Image
                    modelnames2 = []
                    modelnames2.append('modeled')
                    modelnames2.append('observed')
                    plot_model  = plt.plot(obs_results['time'], obs_results[name0]) 
                    plot_obs    = plt.plot(obs_results['time'][idwanted], obs_results[name1][idwanted], '.k')

                    # Adjust layout and show
                    plt.ylabel(name2)
                    plt.legend([plot_model[0], plot_obs[0]], modelnames2, loc = "upper left")
                    plt.title(ids[index2])
                    text_wanted = 'RMSE: ' + format(RMSE, '.2f')  + ', MAE: ' + format(MAE, '.2f') + ', bias: ' + format(bias, '.2f') + ', SCI: ' + format(SCI, '.2f')
                    plt.text(0.95, 0.05, text_wanted, horizontalalignment='right', verticalalignment='center',transform = ax.transAxes)
                    plt.xlim([datetime(2024,9,25), datetime(2024,9,28)])

                    # Fixed y axis
                    if option == 0:
                        plt.ylim([0, 10])
                    elif option == 1:
                        plt.ylim([2, 12])
                    elif option == 2:
                        plt.ylim([0, 360])
                    elif option == 3:
                        plt.ylim([10, 90])

                    # Subfigure with location
                    sub_ax = fig.add_axes([0.70, 0.67, 0.2, 0.2],projection=ccrs.PlateCarree())
                    sub_ax.set_extent([-100, -75, 15.411319,32])
                    sub_ax.add_feature(cfeature.LAND)
                    sub_ax.coastlines()
                    track.plot(ax = sub_ax, color = 'k')
                    #track = sub_ax.plot(lineStringObj.xy[0], lineStringObj.xy[1], 'k')
                    sub_ax.plot(df_now2.longitude, df_now2.latitude, '.r')

                    # Print
                    figure_png  = obs_results['name'] + name3 + '.png'
                    figure_png  = os.path.join(destout_model, figure_png)
                    plt.savefig(figure_png, format='png', dpi=300)
                    plt.close()

                
                # # Compare 1D wave spectra
                # try:

                #     # Do this
                #     wanted = []
                #     for row in obs_results['energy_observed']:
                #         first_value = row
                #         if np.isnan(first_value).all():
                #             wanted.append(False)
                #         else:
                #             wanted.append(True)
                    
                #     # Change to boolean
                #     times_wanted            = obs_results['time'][wanted]
                #     filtered_values_obs     = E_smooth[wanted,:]
                #     filtered_values_obs     = np.transpose(np.vstack(filtered_values_obs))
                #     filtered_values_model   = obs_results['energy_modeled'][wanted]
                #     filtered_values_model   = np.transpose(np.vstack(filtered_values_model))

                #     # Propose some axis
                #     values                  = np.sort(filtered_values_model[:])
                #     max_value               = np.shape(values)

                #     # Make figure
                #     fig, ax                 = plt.subplots(2, 1, figsize=(12, 5))
                #     fig.set_size_inches(8.3, 11.7/2)
                #     [X,Y]                   = np.meshgrid(times_wanted, frequency_observed)
                #     ax[0].pcolor(X, Y, filtered_values_obs, cmap='Spectral_r', norm=LogNorm(vmin=1, vmax=100))
                #     ax[0].set_ylim([0.04, 0.5])
                #     ax[0].set_title('observed at ' + obs_results['name'])
                #     ax[0].set_ylabel('frequency [Hz]')
                #     ax[0].set_xticks([])

                #     [X,Y]                   = np.meshgrid(times_wanted, frequency_modeled)
                #     im = ax[1].pcolor(X, Y, filtered_values_model, cmap='Spectral_r', norm=LogNorm(vmin=1, vmax=100))
                #     ax[1].set_ylim([0.04, 0.5])
                #     ax[1].set_title('modeled with HurryWave')
                #     ax[1].set_ylabel('frequency [Hz]')

                #     cbar = plt.colorbar(im, ax=ax)
                #     cbar.set_label('energy density [m$^2$/Hz]')  # Label for the colorbar

                #     # Print
                #     figure_png  = obs_results['name'] + '_spectra.png'
                #     figure_png  = os.path.join(destout, figure_png)
                #     plt.savefig(figure_png, format='png', dpi=300)
                #     plt.close()

                # except:

                #     # Failed 
                #     print(f" => failed making the spectra")
                

                # Compare 2D wave spectra 
                # to do

                # Done with this all
                # Store all wave spectra
                obs_results_all.append(obs_results)
        

        # Done with all the reading and post-processing: compare total skill
        for option, parameter in enumerate(parameters_wanted):

            # Flavors
            if option == 0: name0 = 'height_modeled';       name1 = 'height_observed';      name2 = 'significant wave height ($H_s$) [m]'; name3 = '_Hs'
            if option == 1: name0 = 'period_modeled';       name1 = 'period_observed';      name2 = 'mean wave period ($T_m0$) [s]'; name3 = '_Tm'
            if option == 2: name0 = 'direction_modeled';    name1 = 'direction_observed';   name2 = 'mean wave direction ($D_m$) [$\circ$]'; name3 = '_Dm'
            if option == 3: name0 = 'spreading_modeled';    name1 = 'spreading_observed';   name2 = 'mean wave spreading ($s$) [\circ]'; name3 = '_s'

            # Loop over all obs
            obs  = []; model = []; marker = []; legend_name = []
            for index, obs_results in enumerate(obs_results_all):

                # Combine all the data
                obs.extend( (obs_results[name1].tolist()) )
                model.extend( (obs_results[name0].tolist()) )
                marker.extend( np.full(len((obs_results[name1])),index).tolist())
                legend_name.append(obs_results['name'])

            # Make figure
            fig, ax = plt.subplots() # for landscape
            fig.set_size_inches(8.3, 11.7/2)

            # Subplot 1: Image
            cmap            = plt.cm.get_cmap('turbo',max(marker)+1)
            norm            = plt.Normalize(vmin=min(marker)-1, vmax=max(marker))  # Adjust the range as needed
            plot_model      = plt.scatter(obs, model, 5, marker, 'o', cmap=cmap, norm=norm)
            cbar            = plt.colorbar(plot_model)
            ticks           = np.arange(min(marker)-0.5, max(marker)+0.5, step=1)  # Values from 1 to 13 with steps of 1
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(legend_name)

            # Set specifics
            if option == 0:
                plt.plot([0, 10], [0, 10], '--k')
                plt.xlim([0, 10])
                plt.ylim([0, 10])
                plt.xlabel('$H_s$ - observed [m]')
                plt.ylabel('$H_s$ - modeled [m]')
                plt.title('significant wave height - Hurricane {}'.format(hurricane_name))
                figure_png  = '_overview_Hs.png'

            # Set specifics
            if option == 1:
                plt.plot([2, 12], [2, 12], '--k')
                plt.xlim([2, 12])
                plt.ylim([2, 12])
                plt.xlabel('$T_m$ - observed [s]')
                plt.ylabel('$T_m$ - modeled [s]')
                plt.title('mean period - Hurricane {}'.format(hurricane_name))
                figure_png  = '_overview_Tm.png'

            # Set specifics
            if option == 2:
                plt.plot([0, 360], [0, 360], '--k')
                plt.xlim([0, 360])
                plt.ylim([0, 360])
                plt.xlabel('$D_m$ - observed [$\circ$]')
                plt.ylabel('$D_m$ - modeled [$\circ$]')
                plt.title('mean direction Hurricane {}'.format(hurricane_name))
                figure_png  = '_overview_Dm.png'

            # Set specifics
            if option == 3:
                plt.plot([10, 90], [10, 90], '--k')
                plt.xlim([10, 90])
                plt.ylim([10, 90])
                plt.xlabel('$s$ - observed [$\circ$]')
                plt.ylabel('$s$ - modeled [$\circ$]')
                plt.title('mean spreading - Hurricane {}'.format(hurricane_name))
                figure_png  = '_overview_s.png'

            # Compute skill- Hurrywave
            actual          = obs
            pred            = model
            actual, pred    = np.array(actual), np.array(pred)
            idwanted1       = np.isfinite(actual)
            idwanted2       = np.isfinite(pred) 
            idwanted        = idwanted1 & idwanted2
            actual          = actual[idwanted]
            pred            = pred[idwanted]
            bias            = np.nanmean(actual - pred)
            MAE             = np.nanmean(abs(actual - pred))
            RMSE            = np.sqrt(np.square(np.subtract(actual,pred)).mean())
            SCI             = RMSE/np.nanmean(pred)
            text_wanted = 'RMSE: ' + format(RMSE, '.2f')  + ', MAE: ' + format(MAE, '.2f') + ', bias: ' + format(bias, '.2f') + ', SCI: ' + format(SCI, '.2f')
            plt.text(0.95, 0.05, text_wanted, horizontalalignment='right', verticalalignment='center',transform = ax.transAxes)

            # Print
            figure_png  = os.path.join(destout_model, figure_png)
            plt.savefig(figure_png, format='png', dpi=300)
            plt.close()

    # Done
    print('done')

    # Save all the observational points
    file_path = os.path.join(destout_model, 'data.pkl')
    with open(file_path, "wb") as file:
        pickle.dump(obs_results_all, file)


    # # Wave spectra plotting
    # plot = 0
    # if plot == 1:
    #     # Remove data below certain value
    #     id_low              = spectra_model < bounds[0]
    #     spectra_model[id_low] = np.nan

    #     # Make figure
    #     fig, ax             = plt.subplots(subplot_kw={'projection': 'polar'})
    #     fig.set_size_inches(8.3, 11.7/2)
    #     reversed_cmap       = plt.cm.get_cmap('Spectral_r')
    #     c                   = ax.pcolor(T, R, spectra_model,norm=norm, cmap=reversed_cmap)
    #     cbar                = plt.colorbar(c, extend="max")
    #     ax.grid(True)

    #     # Get some information on this time stamp
    #     htext1              = f'H_s - {obs_results["height_observed"][index]:.1f} - {obs_results["height_modeled"][index]:.1f}' +'\meter' 
    #     htext2              = f'T_m - {obs_results["period_observed"][index]:.1f} - {obs_results["period_modeled"][index]:.1f}' +'\seconds' 
    #     htext3              = f'D_m - {obs_results["direction_observed"][index]:.0f} - {obs_results["direction_modeled"][index]:.0f}' +'\degrees' 

    #     # Print
    #     dt                  = df_now2.index[index]
    #     filename_format     = '%Y%m%d_%H%M%S'  # YearMonthDay_HourMinuteSecond
    #     date_string         = dt.strftime(filename_format)
    #     figure_png          = obs_results['name'] + '_' + date_string + '.png'
    #     figure_png          = os.path.join(destout, figure_png)
    #     plt.savefig(figure_png, format='png', dpi=300)
    #     plt.close()
