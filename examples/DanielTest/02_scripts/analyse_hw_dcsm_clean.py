"""

Script to validate HW results for refraction case (SWIVT a021refra001_00)

Created on 13-02-2024

@author: kvanasselt
"""

#%%
import os
import shutil
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
import os
import xugrid as xu
from datetime import datetime

# %% Paths

column_names = ('time', 'x', 'y', 'Hs', 'Tm_10', 'Tp', 'DIR')
base_dir = r'p:\11204750-hurrywave'
dateparse = lambda x: datetime.strptime(x, '%Y%m%d.%H%M%S')
main_dir = r'p:\11204750-hurrywave\02_modelling\f998dcs13'


#%% Hurrywave

main_dir = r'p:\11204750-hurrywave\02_modelling\f998dcs13'
run_dir = os.path.join(main_dir, '04_modelruns', 'extended_nobnd')
data_dir = os.path.join(main_dir, '01_data')

map = xr.open_dataset(os.path.join(run_dir, 'hurrywave_map.nc'))
his = xr.open_dataset(os.path.join(run_dir, 'hurrywave_his.nc'))
his = his.assign_coords({"stations": [x.decode("utf-8").strip() for x in his.station_name.values]})

print(list(map.keys()))

#%% SWAN

swan_dir = os.path.join(base_dir, r'01_data\00_SWIVT\sessions\session001\SWAN4131AB\f998dcs13001_000_edit\f998dcs13001\model_io\swivt_f998dcs13001_loc.tab')
df_swan = pd.read_csv(swan_dir,
                        sep='\s+', 
                        skiprows = 7,
                        usecols= [0,1,2,4,6,8,9],
                        names = column_names, 
                        parse_dates= [0],
                        date_parser = dateparse,
                        header = None)


 
# Add the stations to the dataframe

df_swan_copy = df_swan
station_names =  his.stations.values

df_swan_copy["stations"] = np.tile(station_names , 120)

# Set the index of the dataframe to be a multi-index using 'time', 'x', and 'y'
df_swan_copy = df_swan_copy.set_index(['time', 'stations'])

# Convert the dataframe to xarray
swan = df_swan_copy.to_xarray()




#%% Observations 

import csv

dir_noordzee_project = os.path.join(main_dir, r'01_data\swannoordzee')
station_names = pd.read_csv(os.path.join(dir_noordzee_project, 'stations.txt'), header = None)
xr_obs = xr.Dataset()

#%%
for station_id, name in enumerate(station_names[0]):

    print(name)
    
    dir_station = os.path.join(dir_noordzee_project, name)

    df_obs_station_H = pd.DataFrame()
    df_obs_station_T = pd.DataFrame()
    df_obs_station_DIR = pd.DataFrame()

    station_files = glob.glob(os.path.join(dir_station, '*.csv'))

    for csv_file in station_files:
        df= pd.read_csv(csv_file, nrows = 3, delimiter= ';', header = None)
        variable = df[1][2].strip()

        if variable == "Tmmin10":
            
            df_obs_new = pd.read_csv(csv_file, delimiter = ';', skiprows = 59, decimal=',')
            df_obs_new['time'] = pd.to_datetime(df_obs_new.datum+ ' '+df_obs_new.tijd, dayfirst=True)
            df_obs_new['Tm-10'] =  df_obs_new['waarde']
            df_obs_new.set_index(['time'], inplace = True)
            df_obs_new = df_obs_new['Tm-10']

            df_obs_station_T = pd.concat([df_obs_station_T, df_obs_new])


        if variable == "Hm0":
            
            df_obs_new = pd.read_csv(csv_file, delimiter = ';', skiprows = 59, decimal=',')
            df_obs_new['time'] = pd.to_datetime(df_obs_new.datum+ ' '+df_obs_new.tijd, dayfirst=True)
            df_obs_new['Hm0'] =  df_obs_new['waarde']
            df_obs_new.set_index(['time'], inplace = True)
            df_obs_new = df_obs_new['Hm0']

            df_obs_station_H = pd.concat([df_obs_station_H, df_obs_new])

            
        if variable == "Th0":
            
            df_obs_new = pd.read_csv(csv_file, delimiter = ';', skiprows = 59, decimal=',')
            df_obs_new['time'] = pd.to_datetime(df_obs_new.datum+ ' '+df_obs_new.tijd, dayfirst=True)
            df_obs_new['DIR'] =  df_obs_new['waarde']
            df_obs_new.set_index(['time'], inplace = True)
            df_obs_new = df_obs_new['DIR']

            df_obs_station_DIR = pd.concat([df_obs_station_DIR, df_obs_new])

        else:
            continue

    df_obs_station_T = df_obs_station_T.rename(columns = {0:"Tm-10"}) 
    df_obs_station_H = df_obs_station_H.rename(columns = {0:"Hm0"}) 
    df_obs_station_DIR = df_obs_station_DIR.rename(columns = {0:"DIR"}) 

    #print(df_obs_station_T)

    # Drop duplicate dates in the dataset
     
    # Check for duplicate indices in df_obs_station_H and df_obs_station_T
    duplicate_indices_H = df_obs_station_H.index.duplicated()
    duplicate_indices_T = df_obs_station_T.index.duplicated()
    duplicate_indices_DIR = df_obs_station_DIR.index.duplicated()

    # If there are any duplicate indices, handle them accordingly (e.g., drop duplicates)
    if any(duplicate_indices_H) or any(duplicate_indices_T):
        df_obs_station_H = df_obs_station_H[~duplicate_indices_H]
        df_obs_station_T = df_obs_station_T[~duplicate_indices_T]
        df_obs_station_DIR = df_obs_station_DIR[~duplicate_indices_DIR]


    df_obs_station = pd.concat([df_obs_station_H, df_obs_station_T, df_obs_station_DIR], axis = 1)
    df_obs_station["station"] = f'station_{station_id + 1:03d}'
    df_obs_station.index.rename("time", inplace = True)

    df_obs_station.set_index('station',append=True, inplace = True)

    xr_obs_station = df_obs_station.to_xarray()

    if station_id > 0:
        xr_obs = xr.concat([xr_obs, xr_obs_station], dim='station')
    else:
        xr_obs = xr_obs_station


# %% Save as NetCDF

xr_obs.to_netcdf(os.path.join(dir_noordzee_project, "obs_data_2013_new.nc"))
xr_obs.close()

#%% Get NetCDF and plot

import scipy
from sklearn.metrics import mean_squared_error

xr_obs_new = xr.open_dataset(os.path.join(dir_noordzee_project, "obs_data_2013_new.nc"))

idx = [0,1,2,3,5,6,7,8]
stations = his.stations.values[idx]

time_period = slice("2013-12-04", "2013-12-07")

#%%

#Wave height

for ix, station in enumerate(stations):

    fig, ax = plt.subplots(3,1, figsize = (15, 10))

    station_name = station_names[0][ix]

    (swan.sel(stations=station)['Hs']).plot(label=f'SWAN {station_name}',
                                                            ax = ax[0], 
                                                            color='red')
    (xr_obs_new.sel(station=station)['Hm0']/100).plot.scatter(label=f'Observations {station_name}',
                                                            ax = ax[0], 
                                                            color='grey')
    
    (his.sel(stations=station)['point_hm0']).plot(label=f"HW_{station}", 
                                                  ax = ax[0],
                                                  color= "navy")
    


    
    try:
        y_actual = xr_obs_new.sel(station=station, time = time_period)['Hm0']/100
        y_predict = his.sel(stations=station, time = time_period)['point_hm0'].resample(time= '10Min').mean()

        outlier_value = 100
        mask_outliers = y_actual < outlier_value

        y_actual = y_actual[mask_outliers]
        y_predict = y_predict[mask_outliers]

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_actual, y_predict)
        rmse = mean_squared_error(y_actual, y_predict, squared=False)

        r_squared = r_value**2

        ax[0].text(x = 0.15, y = 0.95 , s = f"$R^2$ = {r_squared:.3f}", 
                fontsize = 12,
                ha='left', va='top', 
                transform=ax[0].transAxes)

        ax[0].text(x = 0.15, y = 0.80 , s = f"$RMSE$ = {rmse:.3f}", 
                fontsize = 12,
                ha='left', va='top', 
                transform=ax[0].transAxes)
    except ValueError:
        print(f"No wave height data available for station {station_name}")

    ax[0].grid()
    ax[0].legend(loc='upper left')

    ax[0].set_xlim(datetime(2013,12,4), datetime(2013,12,8))
    ax[0].set_ylim(0, 12)
    ax[0].set_title('Significant wave height')


    # Wave period

    (swan.sel(stations=station)['Tm_10']).plot(label=f"SWAN_{station}", ax = ax[1], color='red')

    (xr_obs_new.sel(station=station)['Tm-10']).plot.scatter(label=f'Observations {station_name}', 
                                                        ax = ax[1],
                                                        color='grey')

    (his.sel(stations=station)['point_tp']/1.22).plot(label=f"HW_{station}", ax = ax[1], color= "navy")

   
    try:

        y_actual = xr_obs_new.sel(station=station, time = time_period)['Tm-10']
        y_predict = (his.sel(stations=station, time = time_period)['point_tp']/1.22).resample(time= '10Min').mean()

        outlier_value = 100
        mask_outliers = y_actual < outlier_value

        y_actual = y_actual[mask_outliers]
        y_predict = y_predict[mask_outliers]

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_actual, y_predict)
        rmse = mean_squared_error(y_actual, y_predict, squared=False)

        r_squared = r_value**2

        ax[1].text(x = 0.05, y = 0.95 , s = f"$R^2$ = {r_squared:.3f}", 
                    fontsize = 12,
                    ha='left', va='top', 
                    transform=ax[1].transAxes)

        ax[1].text(x = 0.05, y = 0.80 , s = f"$RMSE$ = {rmse:.3f}", 
                    fontsize = 12,
                    ha='left', va='top', 
                    transform=ax[1].transAxes)
        
    except ValueError:
        print(f"No wave period data available for station {station_name}")

    # Wave direction

    (xr_obs_new.sel(station=station)['DIR']).plot.scatter(label=f'Observations {station_name}', 
                                                        ax = ax[2], 
                                                        color='grey')
    
    (swan.sel(stations=station)['DIR']).plot(label=f"SWAN_{station}", ax = ax[2], color='red')
    (his.sel(stations=station)['point_wavdir']).plot(label=f"HW_{station}", ax = ax[2], color= "navy")

    try:
        y_actual = xr_obs_new.sel(station=station, time = time_period)['DIR']
        y_predict = his.sel(stations=station, time = time_period)['point_wavdir'].resample(time= '10Min').mean()

        outlier_value = 10000
        mask_outliers = y_actual < outlier_value

        y_actual = y_actual[mask_outliers]
        y_predict = y_predict[mask_outliers]

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_actual, y_predict)
        rmse = mean_squared_error(y_actual, y_predict, squared=False)

        r_squared = r_value**2

        ax[2].text(x = 0.05, y = 0.95 , s = f"$R^2$ = {r_squared:.3f}", 
                fontsize = 12,
                ha='left', va='top', 
                transform=ax[2].transAxes)

        ax[2].text(x = 0.05, y = 0.80 , s = f"$RMSE$ = {rmse:.3f}", 
                fontsize = 12,
                ha='left', va='top', 
                transform=ax[2].transAxes)
    
    except ValueError:
        print(f"No wave direction data available for station {station_name}")

    



    ax[1].grid()
    #ax[1].legend(bbox_to_anchor=[1, 1], loc='upper left')

    ax[1].set_xlim(datetime(2013,12,4), datetime(2013,12,8))
    ax[1].set_ylim(4, 16)
    ax[1].set_title('Wave period (Tm-10)')

    ax[2].grid()
    #ax[2].legend(bbox_to_anchor=[1, 1], loc='upper left')

    ax[2].set_xlim(datetime(2013,12,4), datetime(2013,12,8))
    ax[2].set_ylim(200, 370)
    ax[2].set_title('Wave direction')

    
    save_dir = r'p:\11204750-hurrywave\03_reporting\testbed\fig\f998dcs13'

    fig.suptitle(f"{station}", fontsize = 20)

    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, f"{station_name}.png"))


# %% Scatter plots

idx = [0,1,2,3,5,6,7,8]

stations = his.stations.values[idx]

fig, ax = plt.subplots(2,1, figsize = (10,20))

ax[0].plot([0, 1000], [0, 1000], color = 'black', linestyle = '--')
ax[1].plot([0, 1000], [0, 1000], color = 'black', linestyle = '--')

cmap  = plt.cm.get_cmap('turbo', len(stations))

y_actual_all_H = []
y_predict_all_H = []
y_actual_all_T = []
y_predict_all_T = []



for ix, station in enumerate(stations):

    station_name = station_names[0][ix]
    # (swan.sel(stations=station)['Hs']).plot(label=f'SWAN {station_name}',


    y_actual = xr_obs_new.sel(station=station, time = time_period)['Hm0']/100
    y_predict = his.sel(stations=station, time = time_period)['point_hm0'].resample(time= '10Min').mean()

    outlier_value = 100
    mask_outliers = y_actual < outlier_value

    y_actual = y_actual[mask_outliers]
    y_predict = y_predict[mask_outliers]

    y_actual_all_H = np.concatenate((y_actual_all_H, y_actual.values))
    y_predict_all_H = np.concatenate((y_predict_all_H, y_predict.values))

    # scatter plot this data with a color representing the station in a certain colormap

    ax[0].scatter(y_actual.data, y_predict.data, label = f"{station}", color = cmap(ix))

    #ax[0].scatter(y_actual.data, y_predict.data, label = f"{station_name}", c = ,cmap = cmap)

    y_actual = xr_obs_new.sel(station=station, time = time_period)['Tm-10']
    y_predict = (his.sel(stations=station, time = time_period)['point_tp']/1.22).resample(time= '10Min').mean()

    outlier_value = 100
    mask_outliers = y_actual < outlier_value

    y_actual = y_actual[mask_outliers]
    y_predict = y_predict[mask_outliers]

    ax[1].scatter(y_actual.data, y_predict.data, label = f"{station}", color = cmap(ix))

    y_actual_all_T = np.concatenate((y_actual_all_T, y_actual.values))
    y_predict_all_T = np.concatenate((y_predict_all_T, y_predict.values))

    # y_actual = y_actual[mask_outliers]
    # y_predict = y_predict[mask_outliers]


    # ax[0].text(x = 0.15, y = 0.95 , s = f"$R^2$ = {r_squared:.3f}", 
    #            fontsize = 12,
    #            ha='left', va='top', 
    #            transform=ax[0].transAxes)

    # ax[0].text(x = 0.15, y = 0.80 , s = f"$RMSE$ = {rmse:.3f}", 
    #            fontsize = 12,
    #            ha='left', va='top', 
    #            transform=ax[0].transAxes)


    
ax[0].grid()
ax[0].legend(loc='upper left')

#ax[0].set_xlim(datetime(2013,12,4), datetime(2013,12,8))
ax[0].set_ylim(0, 12)
ax[0].set_xlim(0, 12)
ax[0].set_xlabel('Observed $H_s$ [m]', fontsize = 12)
ax[0].set_ylabel('Modelled $H_s$ [m]', fontsize = 12)
ax[0].set_title('Significant wave height')

ax[1].grid()
ax[1].legend(loc='upper left')

#ax[0].set_xlim(datetime(2013,12,4), datetime(2013,12,8))
ax[1].set_ylim(3, 30)
ax[1].set_xlim(3, 30)
ax[1].set_xlabel('Observed $T_{m-10}$ [m]', fontsize = 12)
ax[1].set_ylabel('Modelled $Tm{-10}$ [m]', fontsize = 12)
ax[1].set_title('Wave period (Tm-10)')


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_actual_all_H, y_predict_all_H)
rmse = mean_squared_error(y_actual_all_H, y_predict_all_H, squared=False)

print(r_value)
print(rmse)

r_squared = r_value**2

ax[0].text(x = 0.05, y = 0.72 , s = f"$R^2$ = {r_squared:.3f}",
              fontsize = 12,
              ha='left', va='top', 
              transform=ax[0].transAxes)


ax[0].text(x = 0.05, y = 0.69 , s = f"$RMSE$ = {rmse:.3f}",
                fontsize = 12,
                ha='left', va='top', 
                transform=ax[0].transAxes)


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_actual_all_T, y_predict_all_T)
rmse = mean_squared_error(y_actual_all_T, y_predict_all_T, squared=False)

print(r_value)
print(rmse)


r_squared = r_value**2

ax[1].text(x = 0.05, y = 0.72 , s = f"$R^2$ = {r_squared:.3f}",
              fontsize = 12,
              ha='left', va='top', 
              transform=ax[1].transAxes)


ax[1].text(x = 0.05, y = 0.69 , s = f"$RMSE$ = {rmse:.3f}",
                fontsize = 12,
                ha='left', va='top', 
                transform=ax[1].transAxes)



save_dir = r'p:\11204750-hurrywave\03_reporting\testbed\fig\f998dcs13'

fig.savefig(os.path.join(save_dir, f"all_stations.png"))



# %%
