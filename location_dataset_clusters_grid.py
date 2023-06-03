import pandas as pd
import numpy as np
import torch
import math
from sklearn.model_selection import train_test_split
from location_dataset import LocationDatasetTrain, LocationDatasetTest
from world import *
import sys

class LocationDatasetDateTimeComponentsGridCluster(object):
    def __init__(self, dataset_csv, seq_len_train, seq_len_test) -> None:
        df = pd.read_csv(dataset_csv)
        time_df = pd.DataFrame()

        df.sort_values(by=['user', 'time'], inplace=True)
        df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]
        df.reset_index(drop=True, inplace=True)

        # Convert the location id (a hash) to a location index
        unique_locations = df['location'].unique()
        self.unique_locations_number = len(unique_locations)
        location_id_map = {location_id: i for i, location_id in enumerate(unique_locations)}
        df['location_index'] = df['location'].apply(lambda x: location_id_map[x])

        print('Generating grid...')
        grid: list = generate_grid(df['latitude'].min(), df['longitude'].min(), df['latitude'].max(), df['longitude'].max())
        print(grid)
        df['grid_index'] = df[['latitude', 'longitude']].apply(lambda row: get_grid_index(row['latitude'], row['longitude'], grid), axis=1)
        print('Grid generated.')

        timestamp_series = pd.to_datetime(df['time'], unit='s')
        time_df['week_day'] = timestamp_series.dt.weekday

        hours = [(i, i + 3) for i in range(0, 21, 4)]
        time_df['hour'] = timestamp_series.dt.hour
        time_df['hour'] = time_df['hour'].apply(lambda x: get_hour_index(x, hours))
        time_df['day'] = timestamp_series.dt.day
        time_df['month'] = timestamp_series.dt.month
        time_df = pd.get_dummies(time_df, columns=['week_day', 'hour', 'day', 'month'], prefix=['week_day', 'hour', 'day', 'month']).astype(np.int32)

        df.drop(columns=['location', 'location_index', 'latitude', 'longitude', 'time'], inplace=True)

        user_df = pd.get_dummies(df['user'], prefix='user').astype(np.int32)

        unique_grids = df['grid_index'].unique()
        unique_grids_map = {grid_index: i for i, grid_index in enumerate(unique_grids)}
        df['grid_index'] = df['grid_index'].apply(lambda x: unique_grids_map[x])
        one_hot_grid = pd.get_dummies(df['grid_index'], prefix='grid_index').astype(np.int32)
        result_df = pd.concat([user_df, time_df, one_hot_grid], axis=1)

        print('Generating train and test sets...')
        X = result_df.iloc[:, :(len(time_df.iloc[0]) + len(user_df.iloc[0]))]
        y = result_df.iloc[:, (len(time_df.iloc[0]) + len(user_df.iloc[0])):]

        users = df['user'].unique()
        users.sort()

        train_set, test_set = [], []
        for user in users:
            user_df = pd.concat([X, y], axis=1)
            user_df = user_df[user_df[f'user_{user}'] == 1]
            X_user = user_df.iloc[:, :(len(time_df.iloc[0]) + len(users))]
            y_user = user_df.iloc[:, (len(time_df.iloc[0]) + len(users)):]
            x_train_user, x_test_user, y_train_user, y_test_user = train_test_split(X_user.values, y_user.values, test_size=0.15, shuffle=False)
            train_set.append((x_train_user, y_train_user))
            test_set.append((x_test_user, y_test_user))

        x_train, y_train = np.concatenate([x_train for x_train, _ in train_set]), np.concatenate([y_train for _, y_train in train_set])
        x_test, y_test = np.concatenate([x_test for x_test, _ in test_set]), np.concatenate([y_test for _, y_test in test_set])
        print('Train and test sets generated.')

        x_train_tensor = torch.from_numpy(x_train).float()
        x_test_tensor = torch.from_numpy(x_test).float()

        y_train_tensor = torch.from_numpy(y_train).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        self.train_set: LocationDatasetTrain = LocationDatasetTrain(x_train_tensor, y_train_tensor, seq_len_train)
        self.test_set: LocationDatasetTest = LocationDatasetTest(x_test_tensor, y_test_tensor, seq_len_test)   


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def generate_grid(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> list:
    latitudes = np.linspace(min_lat, max_lat, num=GRID_SIZE + 1)
    longitudes = np.linspace(min_lon, max_lon, num=GRID_SIZE + 1)
    grid = []
    print(haversine_distance(latitudes[0], longitudes[0], latitudes[1], longitudes[1]))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect_min_lat = latitudes[i]
            rect_max_lat = latitudes[i + 1]
            rect_min_lon = longitudes[j]
            rect_max_lon = longitudes[j + 1]
            grid.append((rect_min_lat, rect_min_lon, rect_max_lat, rect_max_lon))

    return np.array(grid)


def get_grid_index(latitude: float, longitude: float, grid: np.array) -> int:
    condition = (grid[:, 0] <= latitude) & (latitude <= grid[:, 2]) & (grid[:, 1] <= longitude) & (longitude <= grid[:, 3])
    indices = np.where(condition)[0]
    if len(indices) == 0:
        return -1
    return indices[0]


def get_hour_index(hour: int, hours: list) -> int:
    for index, (min_hour, max_hour) in enumerate(hours):
        if min_hour <= hour <= max_hour:
            return index
    return -1
