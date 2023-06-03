import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from location_dataset import LocationDatasetTrain, LocationDatasetTest


class LocationDatasetScaledTimestamps(object):
    def __init__(self, dataset_csv, seq_len) -> None:
        df = pd.read_csv(dataset_csv)

        df.sort_values(by=['time'], inplace=True)
        self.seq_len = seq_len
        df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]

        # One-hot encode the location
        unique_locations = df['location'].unique()
        self.unique_locations_number = len(unique_locations)
        location_id_map = {location_id: i for i, location_id in enumerate(unique_locations)}

        df['location_index'] = df['location'].apply(lambda x: location_id_map[x])

        one_hot = np.zeros((len(df), len(unique_locations)))
        one_hot[np.arange(len(df)), df['location_index']] = 1

        scaler = StandardScaler()
        df.insert(1, 'scaled_time', scaler.fit_transform(df[['time']]))
        # df.insert(2, 'scaled_latitude', scaler.fit_transform(df[['latitude']]))
        # df.insert(3, 'scaled_longitude', scaler.fit_transform(df[['longitude']]))

        df.drop(['location', 'user', 'latitude', 'longitude', 'time'], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)

        one_hot_locations = pd.get_dummies(df['location_index'], prefix='location_index').astype(np.int32)
        df.drop(['location_index'], axis=1, inplace=True)
        df = pd.concat([df, one_hot_locations], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :1].values, df.iloc[:, 1:].values, test_size=0.2, shuffle=False)
        x_train_tensor = torch.from_numpy(x_train).float()
        x_test_tensor = torch.from_numpy(x_test).float()

        y_train_tensor = torch.from_numpy(y_train).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        self.train_set: LocationDatasetTrain = LocationDatasetTrain(x_train_tensor, y_train_tensor, seq_len)
        self.test_set: LocationDatasetTest = LocationDatasetTest(x_test_tensor, y_test_tensor, seq_len)
