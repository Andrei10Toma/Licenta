import os
import math

CLUSTERING_DISTANCE = 3.0
GRID_SIZE = 1280
# GRID_SIZE = 2560
# GRID_SIZE = 10
# PREPROCESS_DATA = True
PREPROCESS_DATA = False

BRIGHTKITE_RAW = os.path.join('dataset', 'brightkite', 'loc-brightkite_totalCheckins.txt')
BRIGHTKITE = os.path.join('dataset', 'brightkite', 'loc-brightkite_totalCheckins.txt.csv')
BRIGHTKITE_MODEL = os.path.join('dataset', 'brightkite', 'loc-brightkite_totalCheckins_model.txt.csv')
BRIGHTKITE_NAME = 'brightkite'

GOWALLA_RAW = os.path.join('dataset', 'gowalla', 'loc-gowalla_totalCheckins.txt')
GOWALLA = os.path.join('dataset', 'gowalla', 'loc-gowalla_totalCheckins.txt.csv')
GOWALLA_NAME = 'gowalla'

YELP_BUSINESS = os.path.join('dataset', 'yelp', 'yelp_academic_dataset_business.json')
YELP_CHECKIN = os.path.join('dataset', 'yelp', 'yelp_academic_dataset_checkin.json')
YELP_REVIEW = os.path.join('dataset', 'yelp', 'yelp_academic_dataset_review.json')
YELP_USER = os.path.join('dataset', 'yelp', 'yelp_academic_dataset_user.json')
YELP_NAME = 'yelp'

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
