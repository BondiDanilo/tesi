#!/usr/bin/env python
# coding: utf-8

# In[]: Packages
import numpy as np

import matplotlib.pyplot as plt

# import Cartopy Coordinate Reference System
import cartopy.crs as ccrs

# Import features from Cartopy (country outlines, rivers, etc)
import cartopy.feature as cfeat

# Read CSV format files
import pandas as pd

from scipy.spatial import ConvexHull, convex_hull_plot_2d

from sklearn.cluster import KMeans
# In[]: Import data
n_points = 1+100

file_path = "../data/lx_data_20200711.csv"
dataset = pd.read_csv(file_path)

if n_points > dataset.size:
    n_points = dataset.size
    print("WARNING: Chosen number of point exceeds the size of data. Resized variable 'n_points' accordingly.")
lx_data = pd.DataFrame(data=dataset[1:n_points], columns=["latitude","longitude","time_utc"])

lon = lx_data["longitude"]
lat = lx_data["latitude"]

# In[ ]: KMeans Clustering
X = np.column_stack([lon,lat])

n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters).fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

Y = np.column_stack([lat, lon, labels])
cluster_data = pd.DataFrame(data=Y,columns=["longitude","latitude","label"])

# In[] Plot clusters with different colors
#print(cluster_data)
color=['r','g','b']
figura = plt.figure(1)
for i in range(3):
    xx = cluster_data.loc[cluster_data['label']==i] 
    plt.plot(xx["longitude"],xx["latitude"],'.',c=color[i])
    
# In[]: Convex Hull of a cluster
xx = cluster_data.loc[cluster_data['label']==1]
yy = np.column_stack([xx["longitude" ],xx["latitude"]])

hull = ConvexHull(yy)       
plt.plot(yy[:,0],yy[:,1],'.')

for index in hull.vertices:
    print(hull.points[index])

for simplex in hull.simplices:
    plt.plot(yy[simplex, 0], yy[simplex, 1], 'k-')
    
# In[]: Plot map

# Boundaries of the map
bound_padding = 0.5
west_bound = min(lon) - bound_padding
east_bound = max(lon) + bound_padding
north_bound = max(lat) + bound_padding
south_bound = min(lat) - bound_padding
lon_center = np.mean(lon)
lat_center = np.mean(lat)

# First create a figure
fig_width = 10
fig_height = 8
fig = plt.figure(figsize=(fig_width,fig_height))
# figsize in inches

# we need to add a projection since our plot is a map. 
# The CCRS package has available the Lambert Conformal Projection
proj = ccrs.LambertConformal(lon_center,lat_center)
#proj = ccrs.PlateCarree(10)

# Add an axis with the map
ax = fig.add_subplot(1,1,1, projection=proj) # args: =(n_rows, n_columns, index, opts)

ax.add_feature(cfeat.LAND)
ax.add_feature(cfeat.OCEAN)
ax.add_feature(cfeat.COASTLINE)
ax.add_feature(cfeat.BORDERS, linestyle=':')
ax.add_feature(cfeat.LAKES, alpha=0.5)
ax.add_feature(cfeat.RIVERS)

# set the boundary of the plot ([W_lon,E_long,S_lat,N_lat]) 
ax.set_extent([west_bound,east_bound,south_bound,north_bound])

milan = [9.1900,45.4642]

ax.scatter(lon,lat, transform=ccrs.PlateCarree(), marker='.',s=10,c='r')
# The option transform apply a coordinate tranformation, 
# in this case PlateCarree maps (lon,lat) to (x,y)
ax.scatter(centroids[:,0],centroids[:,1],transform=ccrs.PlateCarree(), marker = 'x', s=50, c='k')