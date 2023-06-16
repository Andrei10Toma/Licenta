import pandas as pd
from datetime import datetime
from world import *
import argparse
import json
import numpy as np
import sys

parser = argparse.ArgumentParser(description='Preprocess Human Mobility Dataset')
parser.add_argument('--dataset', '-d', type=str, help='Dataset to preprocess', required=True)
parser.add_argument('--preprocess', '-p', type=bool, help='Preprocess dataset')
parser.add_argument('--min_checkins', '-m', type=int, help='Minimum number of checkins per user', default=1)

args = parser.parse_args()

def convert_txt_to_csv(dataset):
    with open(dataset, 'r') as f:
        data = f.readlines()
        data = [  d.strip().split('\t') for d in data if int(d.strip().split('\t')[0]) < 10 ]
        data = [ [ int(d[0]), datetime.strptime(d[1], "%Y-%m-%dT%H:%M:%SZ").timestamp(), float(d[2]), float(d[3]), d[4] ] for d in data ]
        data = pd.DataFrame(data, columns=[ 'user', 'time', 'latitude', 'longitude', 'location' ])
        data = remove_users_with_less_checkins(data)
        print(f"num_users={len(data['user'].unique())}")
        data.to_csv(BRIGHTKITE, index=False)
        if bool(args.preprocess):
            data = preprocess_dataframe(data)
        data.to_csv(BRIGHTKITE_MODEL, index=False)


def preprocess_dataframe(df_copy: pd.DataFrame):
    df = pd.DataFrame.copy(df_copy, deep=True)
    time_df = pd.DataFrame()

    df.sort_values(by=['user', 'time'], inplace=True)
    df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]
    df.reset_index(drop=True, inplace=True)

    # Convert the location id (a hash) to a location index
    unique_locations = df['location'].unique()
    location_id_map = {location_id: i for i, location_id in enumerate(unique_locations)}
    df['location_index'] = df['location'].apply(lambda x: location_id_map[x])

    print('Generating grid...')
    grid: list = generate_grid(df['latitude'].min(), df['longitude'].min(), df['latitude'].max(), df['longitude'].max())
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
    return result_df


def remove_users_with_less_checkins(df: pd.DataFrame):
    user_checkins = df.groupby('user').size().reset_index(name='count')
    user_checkins = user_checkins[user_checkins['count'] >= args.min_checkins]
    df = df.merge(user_checkins, on='user', how='inner')
    df.drop(columns=['count'], inplace=True)
    unique_users = df['user'].unique()
    unique_users_map = {user: i for i, user in enumerate(unique_users)}
    df['user'] = df['user'].apply(lambda x: unique_users_map[x])
    return df


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


def convert_yelp_to_csv():
    with open(YELP_USER, 'r') as f:
        valid_users = set()
        for _, line in enumerate(f):
            data = json.loads(line)
            data.pop('friends')
            if data['review_count'] > 100:
                pass

if __name__ == '__main__':
    if str(args.dataset).lower() == BRIGHTKITE_NAME:
        convert_txt_to_csv(BRIGHTKITE_RAW)
    elif str(args.dataset).lower() == GOWALLA_NAME:
        convert_txt_to_csv(GOWALLA_RAW)
    elif str(args.dataset).lower() == YELP_NAME:
        convert_yelp_to_csv()
    else:
        print(f'Unknown dataset: {args.dataset}')
