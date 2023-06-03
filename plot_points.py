import matplotlib.pyplot as plt
import pandas as pd
from world import *
from sklearn.cluster import KMeans

df = pd.read_csv(BRIGHTKITE)
df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]
latitude = df['latitude']
longitude = df['longitude']

# data = df[['latitude', 'longitude']].values
# kmeans = KMeans(n_clusters=8).fit(data)
# clusters = kmeans.fit_predict(data)

# plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='rainbow')
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.title('Brightkite Clusters')
# plt.show()

plt.scatter(latitude, longitude)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Brightkite Clusters')
plt.show()
