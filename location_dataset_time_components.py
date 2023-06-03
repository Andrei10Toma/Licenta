import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from location_dataset import LocationDatasetTrain, LocationDatasetTest


class LocationDatasetDateTimeComponents(object):
    def __init__(self, dataset_csv, seq_len_train, seq_len_test) -> None:
        df = pd.read_csv(dataset_csv)
        time_df = pd.DataFrame()

        df.sort_values(by=['time'], inplace=True)
        df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]
        df.reset_index(drop=True, inplace=True)

        unique_locations = df['location'].unique()
        self.unique_locations_number = len(unique_locations)
        location_id_map = {location_id: i for i, location_id in enumerate(unique_locations)}
        df['location_index'] = df['location'].apply(lambda x: location_id_map[x])

        one_hot_locations = pd.get_dummies(df['location_index'], prefix='location').astype(np.int32)
        df = pd.concat([df, one_hot_locations], axis=1)

        timestamp_series = pd.to_datetime(df['time'], unit='s')
        time_df['week_day'] = timestamp_series.dt.weekday
        time_df['hour'] = timestamp_series.dt.hour
        time_df['minute'] = timestamp_series.dt.minute
        time_df = pd.get_dummies(time_df, columns=['week_day', 'hour', 'minute'], prefix=['day', 'hour', 'minute']).astype(np.int32)
        df.drop(columns=['location', 'user', 'latitude', 'longitude', 'time', 'location_index'], inplace=True)

        X = time_df.values
        y = df.values

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        x_train_tensor = torch.from_numpy(x_train).float()
        x_test_tensor = torch.from_numpy(x_test).float()

        y_train_tensor = torch.from_numpy(y_train).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        self.train_set: LocationDatasetTrain = LocationDatasetTrain(x_train_tensor, y_train_tensor, seq_len_train)
        self.test_set: LocationDatasetTest = LocationDatasetTest(x_test_tensor, y_test_tensor, seq_len_test)