import pandas as pd
import numpy as np
import torch
import math
from sklearn.model_selection import train_test_split
from location_dataset import LocationDatasetTrain, LocationDatasetTest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from world import *
import sys

class LocationDatasetDateTimeComponentsKMeansCluster(object):
    def __init__(self, dataset_csv, seq_len_train, seq_len_test) -> None:
        df = pd.read_csv(dataset_csv)
        time_df = pd.DataFrame()

        df.sort_values(by=['time'], inplace=True)
        df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]
        df.reset_index(drop=True, inplace=True)

        # Convert the location id (a hash) to a location index
        unique_locations = df['location'].unique()
        self.unique_locations_number = len(unique_locations)
        location_id_map = {location_id: i for i, location_id in enumerate(unique_locations)}
        df['location_index'] = df['location'].apply(lambda x: location_id_map[x])

        lat_lon_dict = {}

        # Calculate the distance between each location
        for _, row in df[['location_index', 'latitude', 'longitude']].iterrows():
            lat_lon_dict[row['location_index']] = (row['latitude'], row['longitude'])
        distance_dict = {location_index: [ 0 for _ in range(self.unique_locations_number) ] for location_index in lat_lon_dict.keys()}
        for location_index_i, (lat_i, lon_i) in lat_lon_dict.items():
            for location_index_j, (lat_j, lon_j) in lat_lon_dict.items():
                distance_dict[location_index_i][int(location_index_j)] = haversine_distance(lat_i, lon_i, lat_j, lon_j)

        points = list(distance_dict.keys())
        distance_matrix = np.zeros((len(points), len(points)))
        for i, point1 in enumerate(points):
            for j, point2 in enumerate(points):
                if point2 in distance_dict[point1]:
                    distance_matrix[i, j] = distance_dict[point1][int(point2)]
                else:
                    distance_matrix[i, j] = distance_dict[point2][int(point1)]

        # KMeans elbow method
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(distance_matrix)
        labels = kmeans.labels_

        df['location_index'] = df['location_index'].apply(lambda x: labels[x])

        timestamp_series = pd.to_datetime(df['time'], unit='s')
        time_df['week_day'] = timestamp_series.dt.weekday
        time_df['hour'] = timestamp_series.dt.hour
        time_df['minute'] = timestamp_series.dt.minute
        time_df['day'] = timestamp_series.dt.day
        time_df['month'] = timestamp_series.dt.month
        time_df = pd.get_dummies(time_df, columns=['week_day', 'hour', 'minute', 'day', 'month'], prefix=['week_day', 'hour', 'minute', 'day', 'month']).astype(np.int32)

        one_hot_locations = pd.get_dummies(df['location_index'], prefix='location').astype(np.int32)
        df = pd.concat([df, one_hot_locations], axis=1)
        df.drop(columns=['location', 'location_index', 'user', 'latitude', 'longitude', 'time'], inplace=True)

        X = time_df.values
        y = df.values

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        x_train_tensor = torch.from_numpy(x_train).float()
        x_test_tensor = torch.from_numpy(x_test).float()

        y_train_tensor = torch.from_numpy(y_train).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        self.train_set: LocationDatasetTrain = LocationDatasetTrain(x_train_tensor, y_train_tensor, seq_len_train)
        self.test_set: LocationDatasetTest = LocationDatasetTest(x_test_tensor, y_test_tensor, seq_len_test)   


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if lat1 == lat2 and lon1 == lon2:
        return 0.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    r = 6371  # Radius of the Earth in kilometers
    distance = r * c

    return distance
