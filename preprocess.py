import pandas as pd
import os
from datetime import datetime
from world import *

def convert_to_csv(dataset):
    with open(dataset, 'r') as f:
        data = f.readlines()
        data = [  d.strip().split('\t') for d in data if int(d.strip().split('\t')[0]) <= 1 ]
        data = [ [ int(d[0]), datetime.strptime(d[1], "%Y-%m-%dT%H:%M:%SZ").timestamp(), float(d[2]), float(d[3]), d[4] ] for d in data ]
        data = pd.DataFrame(data, columns=[ 'user', 'time', 'latitude', 'longitude', 'location' ])
        data.to_csv(os.path.join(os.path.dirname(dataset), f'{os.path.basename(dataset)}.csv'), index=False)


if __name__ == '__main__':
    convert_to_csv(BRIGHTKITE_RAW)
