import ddlpy
import datetime as dt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")
import os

name = 'Pia'
start_date = dt.datetime(2023, 12, 19)  # Year, Month, Day
end_date = dt.datetime(2023, 12, 23)  # Year, Month, Day

# Set the path to save the data
main_path = r'C:\Users\User\OneDrive\Documents\Python\PYTHON_MSC_CE\Year_2\Python_Thesis\cht_hurrywave\examples\DanielTest\01_data\Waterinfo_RWS'
output_path = os.path.join(main_path, name)
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Get locations available
locations = ddlpy.locations()

# Get all locations with wave data
bool_grootheid = locations['Grootheid.Omschrijving'].str.contains('golf')

# Find these locations
targets = ['north cormorant', 'Platform F16-A', 'K13 Alpha', 'Platform Hoorn Q1-A','Ijmuiden munitiestortplaats', 'Euro platform', 'Platform D15-A']

# Apply filters, first on the wave data, then on the names
wave_data_locs = locations[bool_grootheid]
wave_data_locs = wave_data_locs[wave_data_locs['Naam'].str.lower().isin([name.lower() for name in targets])]

# Get the data
hm0_locations = wave_data_locs[wave_data_locs['Grootheid.Code'].str.contains('Hm0')]
t13_locations = wave_data_locs[wave_data_locs['Grootheid.Code'].str.contains('T1/3')]
tmax_locations = wave_data_locs[wave_data_locs['Grootheid.Code'].str.contains('Tmax')]

# Download the data from the RWS API
def get_data(location, var_name, start_date, end_date, dir_output, dict_measurements,overwrite=True):
    station_id = location.name
    station_messageid = location["Locatie_MessageID"]
    filename = os.path.join(dir_output, f"{station_id}-{station_messageid}.nc")

    if os.path.isfile(filename) and overwrite is False:
        print(f'{station_id}: file already exists and overwrite=False, skipping')
        return

    measurements = ddlpy.measurements(location, start_date=start_date, end_date=end_date)

    if measurements.empty:
        print(f'{station_id}: no measurements found')
        return

    print(f'{station_id}: writing retrieved data to file')

    simplified = ddlpy.simplify_dataframe(measurements)

    station = location['Naam']
    
    dict_measurements[station][var_name] = simplified['Meetwaarde.Waarde_Numeriek']

    print(f'{station}: {var_name} df length : {dict_measurements[station][var_name].shape}')


# Initialize the dictionary
def init_dict_measurements():
    dict_measurements = {}
    for station in wave_data_locs['Naam']:
        dict_measurements[station] = {}
    return dict_measurements

dict_measurements = init_dict_measurements()

# Download and collect Hm0, T13 and Tmax
for _, location in hm0_locations.iterrows():
    get_data(location, 'hm0' ,start_date, end_date,output_path,dict_measurements, overwrite=True)

for _, location in t13_locations.iterrows():
    get_data(location, 't13' ,start_date, end_date,output_path,dict_measurements, overwrite=True)

for _, location in tmax_locations.iterrows():
    get_data(location, 'tmax' ,start_date, end_date,output_path,dict_measurements, overwrite=True)


# Write the data to CSV files, interpolating times
def write_to_csv(dict_measurements, outpath, station):
    # Check which variables are missing or empty, and create NaN series for them
    required_vars = ['hm0', 't13', 'tmax']
    # Determine the length to use for NaNs (use the max length of available variables, or default to 1)
    lengths = [len(dict_measurements[station][v]) for v in dict_measurements[station] if dict_measurements[station][v] is not None and hasattr(dict_measurements[station][v], '__len__')]
    nan_length = max(lengths) if lengths else 1

    # Prepare dataframes for each variable
    dfs = {}
    for var in required_vars:
        if var in dict_measurements[station] and dict_measurements[station][var] is not None and len(dict_measurements[station][var]) > 0:
            df = pd.DataFrame({
                'time': dict_measurements[station][var].index,
                var: dict_measurements[station][var]
            })
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df.loc[~df.index.duplicated(keep='first')]
            dfs[var] = df
        else:
            print(f"{station}: No data for {var}, filling with NaN.")
            # Try to get the index from any available variable, else use a range index
            index = None
            for v in dict_measurements[station]:
                if dict_measurements[station][v] is not None and hasattr(dict_measurements[station][v], 'index'):
                    index = dict_measurements[station][v].index
                    break
            if index is None:
                index = pd.RangeIndex(nan_length)
            df_nan = pd.DataFrame({var: [np.nan]*nan_length}, index=index)
            df_nan.index.name = 'time'
            dfs[var] = df_nan

    # Determine the overall time range for reindexing
    all_indices = [df.index for df in dfs.values() if len(df) > 0]
    if not all_indices:
        print(f"{station}: No data available for any variable.")
        return
    start_time = min(idx.min() for idx in all_indices)
    end_time = max(idx.max() for idx in all_indices)
    hourly_index = pd.date_range(start=start_time, end=end_time, freq='10min')

    # Reindex and interpolate each variable
    for var in required_vars:
        dfs[var] = dfs[var].reindex(hourly_index).interpolate(method='time')

    # Combine all variables into one dataframe
    combined_df = pd.concat([dfs['hm0']['hm0'], dfs['t13']['t13'], dfs['tmax']['tmax']], axis=1)
    combined_df.to_csv(f'{outpath}/{station}.csv', index=True)


# Write all to CSV in outpath
def write_to_csv_all(dict_measurements, outpath):
    for station in dict_measurements.keys():
        write_to_csv(dict_measurements, outpath, station)

write_to_csv_all(dict_measurements, output_path)