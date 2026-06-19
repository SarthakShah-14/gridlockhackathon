import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Bangalore City Center Coordinates
CITY_LAT = 12.9716
CITY_LON = 77.5946

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0 # Radius of earth in kilometers
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate compass bearing from start to end coordinates.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360) % 360

def extract_spatial_features(df: pd.DataFrame, kmeans_model=None, train_mode: bool = True, n_clusters: int = 15):
    df = df.copy()
    
    # 1. Distance from city center
    df['dist_from_city_center'] = haversine_distance(df['latitude'], df['longitude'], CITY_LAT, CITY_LON)
    
    # 2. Distance between start and end coordinates (if available)
    has_end = df['endlatitude'].notnull() & df['endlongitude'].notnull()
    df['route_distance_km'] = 0.0
    if has_end.any():
        df.loc[has_end, 'route_distance_km'] = haversine_distance(
            df.loc[has_end, 'latitude'], df.loc[has_end, 'longitude'],
            df.loc[has_end, 'endlatitude'], df.loc[has_end, 'endlongitude']
        )
    df.loc[~has_end, 'route_distance_km'] = np.nan
    df['route_length'] = df['route_distance_km']
    
    # 3. Midpoints and bearing
    df['mid_latitude'] = np.nan
    df['mid_longitude'] = np.nan
    df['bearing'] = np.nan
    
    if has_end.any():
        df.loc[has_end, 'mid_latitude'] = (df.loc[has_end, 'latitude'] + df.loc[has_end, 'endlatitude']) / 2.0
        df.loc[has_end, 'mid_longitude'] = (df.loc[has_end, 'longitude'] + df.loc[has_end, 'endlongitude']) / 2.0
        df.loc[has_end, 'bearing'] = calculate_bearing(
            df.loc[has_end, 'latitude'], df.loc[has_end, 'longitude'],
            df.loc[has_end, 'endlatitude'], df.loc[has_end, 'endlongitude']
        )
    
    # 3. DBSCAN geographical clustering of (latitude, longitude) (Module 2)
    coords = df[['latitude', 'longitude']].fillna(0)
    
    if train_mode:
        from sklearn.cluster import DBSCAN
        # eps is approx 500 meters (0.0045 degrees), min_samples=5 for Bengaluru incident clusters
        dbscan = DBSCAN(eps=0.0045, min_samples=5)
        labels = dbscan.fit_predict(coords)
        
        # Precompute centroids and sizes
        centroids = {}
        sizes = {}
        unique_labels = set(labels) - {-1}
        for label in unique_labels:
            mask = (labels == label)
            centroids[label] = (coords.loc[mask, 'latitude'].mean(), coords.loc[mask, 'longitude'].mean())
            sizes[label] = int(mask.sum())
        
        dbscan_model = {
            'centroids': centroids,
            'sizes': sizes
        }
    else:
        dbscan_model = kmeans_model  # load passed artifacts dict
        
    centroids = dbscan_model['centroids']
    sizes = dbscan_model['sizes']
    
    # Map points to nearest cluster centroid
    cluster_ids = []
    dist_to_centroids = []
    cluster_sizes = []
    
    for idx, row in coords.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if not centroids:
            cluster_ids.append(-1)
            dist_to_centroids.append(0.0)
            cluster_sizes.append(0)
            continue
            
        min_dist = float('inf')
        best_label = -1
        for label, cent in centroids.items():
            d = haversine_distance(lat, lon, cent[0], cent[1])
            if d < min_dist:
                min_dist = d
                best_label = label
                
        cluster_ids.append(best_label)
        dist_to_centroids.append(min_dist)
        cluster_sizes.append(sizes.get(best_label, 0))
        
    df['geo_cluster'] = cluster_ids
    df['cluster_size'] = cluster_sizes
    df['dist_to_cluster_centroid'] = dist_to_centroids
    df['dist_to_cluster_center'] = dist_to_centroids # fallback alias
    
    return df, dbscan_model
