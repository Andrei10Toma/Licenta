from world import *
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

df = pd.read_csv(BRIGHTKITE)
df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]


