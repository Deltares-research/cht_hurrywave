import numpy as np
def print_variables(data):
    """
    Print the variables in the dataset.
    """
    # (OPTIONAL) Print available parameters in datatory-file
    print(f"{'Variable Name':<15} {'Long Name':<65} {'Units':<15} {'Dimensions':<40}")
    for var_name, data_array in data.variables.items():
        dimensions = str(data_array.dims)
        long_name = data_array.attrs.get('long_name', 'N/A')
        units = data_array.attrs.get('units', 'N/A')
        print(f"{var_name:<15}  {long_name:<65} {units:<15} {dimensions:<40}")


def load_station_data(data, lat_idx, lon_idx):
    """
    Load all relevant variables at a known station grid point (lat_idx, lon_idx).
    """
    variables = {}
    for var_name in data.data_vars:
        if ('valid_time' in data[var_name].dims and
            'latitude' in data[var_name].dims and
            'longitude' in data[var_name].dims):
            variables[var_name] = data[var_name][:, lat_idx, lon_idx].values
    return variables

def load_specific_variables(data, variables):
    """
    Extract and return selected wave-related variables from the grid point data.
    """
    data_dt = np.diff(data.valid_time.values).astype('timedelta64[s]').mean().astype(float)
    data_time = data.valid_time.values

    data_hm0 = variables.get('swh', None)             # Significant wave height
    data_tp = variables.get('pp1d', None)             # Peak period
    data_wavdir = variables.get('mwd', None)          # Mean wave direction
    data_dirspr = variables.get('wdw', None)          # Directional spread
    data_wind_speed = variables.get('shww', None)     # Wind wave height
    data_wind_direction = variables.get('mdww', None) # Wind wave direction

    return data_dt, data_time, data_hm0, data_tp, data_wavdir, data_dirspr, data_wind_speed, data_wind_direction

def get_number_of_latitude_values(data):
    """
    Get the number of latitude values in the dataset.
    """
    return len(data.latitude.values) if 'latitude' in data.dims else 0

def get_number_of_longitude_values(data):
    """
    Get the number of longitude values in the dataset.
    """
    return len(data.longitude.values) if 'longitude' in data.dims else 0

def get_latitude_values(data):
    """
    Get the latitude values from the dataset.
    """
    return data.latitude.values if 'latitude' in data.dims else None

def get_longitude_values(data):
    """
    Get the longitude values from the dataset.
    """
    return data.longitude.values if 'longitude' in data.dims else None

def get_lat_lon_values(data, lat_idx, lon_idx):
    """
    Get the latitude and longitude values for a specific grid point.
    """
    lat = data.latitude.values[lat_idx] if 'latitude' in data.dims else None
    lon = data.longitude.values[lon_idx] if 'longitude' in data.dims else None
    return lat, lon

def get_lat_lon_values_from_data(data):
    """
    Get the latitude and longitude values from the dataset.
    """
    lat = data.latitude.values if 'latitude' in data.dims else None
    lon = data.longitude.values if 'longitude' in data.dims else None
    return lat, lon

def get_closest_lat_idx_from_coords(data, lat):
    """
    Get the closest latitude index from the dataset for a given latitude coordinate.
    """
    lat_values = get_latitude_values(data)
    if lat_values is not None:
        return np.abs(lat_values - lat).argmin()
    return None

def get_closest_lon_idx_from_coords(data, lon):
    """
    Get the closest longitude index from the dataset for a given longitude coordinate.
    """
    lon_values = get_longitude_values(data)
    if lon_values is not None:
        return np.abs(lon_values - lon).argmin()
    return None

def get_closest_lat_lon_idx_from_coords(data, lat, lon):
    """
    Get the closest latitude and longitude indices from the dataset for given coordinates.
    """
    lat_idx = get_closest_lat_idx_from_coords(data, lat)
    lon_idx = get_closest_lon_idx_from_coords(data, lon)
    return lat_idx, lon_idx

def get_closest_lat_lon_from_list_of_coords(data, lat, lon):
    """
    Get the closest latitude and longitude values from the dataset for given coordinates.
    """
    lat_idx, lon_idx = get_closest_lat_lon_idx_from_coords(data, lat, lon)
    lat_values = get_latitude_values(data)
    lon_values = get_longitude_values(data)
    
    if lat_idx is not None and lon_idx is not None:
        return lat_values[lat_idx], lon_values[lon_idx]
    return None, None

def add_stations_to_dict(station_dict, data, lat_idx, lon_idx):
    """
    Add station metadata (name and coordinates) to the dictionary.
    """
    lat, lon = get_lat_lon_values(data, lat_idx, lon_idx)
    
    # Determine the station name
    if station_dict:
        last_station_name = max(station_dict.keys(), key=lambda k: int(k.replace('station', '')))
        station_number = int(last_station_name.replace('station', '')) + 1
    else:
        station_number = 1
    
    station_name = f'station{station_number:03d}'
    
    # Add station metadata to the dictionary
    station_dict[station_name] = (lat, lon)
    return station_dict

def add_stations_to_dict_from_data(station_dict, data):
    """
    Add all stations to the dictionary using the dataset.
    """
    for lat_idx in range(get_number_of_latitude_values(data)):
        for lon_idx in range(get_number_of_longitude_values(data)):
            station_dict = add_stations_to_dict(station_dict, data, lat_idx, lon_idx)
    return station_dict


def add_stations_to_dict_from_coords(station_dict, data, lat, lon):
    """
    Add station data to the dictionary using latitude and longitude coordinates.
    """
    lat_idx, lon_idx = get_closest_lat_lon_idx_from_coords(data, lat, lon)
    return add_stations_to_dict(station_dict, data, lat_idx, lon_idx)

def add_stations_to_dict_from_coordata_list(station_dict, data, coordata_list):
    """
    Add multiple stations to the dictionary using a list of latitude and longitude coordinates.
    """
    for lat, lon in coordata_list:
        station_dict = add_stations_to_dict_from_coords(station_dict, data, lat, lon)
    return station_dict

def create_station_dict(data):
    """
    Create a dictionary of stations with their latitude and longitude as keys.
    """
    station_dict = {}
    station_dict = add_stations_to_dict_from_data(station_dict, data)
    return station_dict

def print_station_dict(station_dict):
    """
    Print the station dictionary.
    """
    print(f"Number of stations in the dictionary: {len(station_dict)}")
    for station_name, (lat, lon) in station_dict.items():
        print(f"{station_name}: Latitude: {lat}, Longitude: {lon}")


def filter_station_dict(station_dict, data, lat_list, lon_list):
    """
    Filter the station dictionary based on the stations closest to the inputted latitudes and longitudes.
    """
    filtered_dict = {}
    for lat, lon in zip(lat_list, lon_list):
        closest_lat, closest_lon = get_closest_lat_lon_from_list_of_coords(data, lat, lon)
        for station_name, (station_lat, station_lon) in station_dict.items():
            if station_lat == closest_lat and station_lon == closest_lon:
                filtered_dict[station_name] = (station_lat, station_lon)
                break
    return filtered_dict

def change_station_name(station_dict, old_name, new_name):
    """
    Change the name of a station in the dictionary.
    """
    if old_name in station_dict:
        station_dict[new_name] = station_dict.pop(old_name)
    else:
        print(f"Station {old_name} not found in the dictionary.")
    return station_dict

def change_all_station_names_to_new_names(station_dict, new_name):
    """
    Change all station names in the dictionary to a new list of names.
    """
    if len(new_name) != len(station_dict):
        raise ValueError("The number of new names must match the number of stations in the dictionary.")
    
    old_names = list(station_dict.keys())
    for old_name, new_name in zip(old_names, new_name):
        station_dict[new_name] = station_dict.pop(old_name)
    return station_dict