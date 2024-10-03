#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:36:53 2024

@author: ericlevenson
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

## Read in maximum lakes shapefile
lakes = gpd.read_file('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/shapefiles/AK25.shp')
lakes = lakes.set_crs('EPSG:3338') # set CRS
## Read in minimum lakes shapefile
drylakes = gpd.read_file('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/shapefiles/AK75.shp')
drylakes = drylakes.set_crs('EPSG:3338') # set CRS
# rename columns
drylakes = drylakes.rename(columns={"area": "dryArea", "perimeter": "dryPerimeter", "geometry": "dryGeometry" })

## Add ID and centroid columns
lakes['id'] = list(range(1, len(lakes)+1))
lakes['centroid'] = str(lakes['geometry'].centroid)

# set geometry and id column
drylakes = drylakes.set_geometry('dryGeometry')
drylakes = drylakes.drop(columns='DN')
drylakes['dryID'] = list(range(1,len(drylakes)+1))


## Add max lake ID to dry lake features using spatial join
drylakes2 = gpd.sjoin(lakes, drylakes, how='right')
# check columns
print(drylakes2.columns)

## Attach dry area, dry perimeter, dry lakeCount to lakes
# get list of max lake ids
idList = lakes['id'].tolist()
# create empty lists for dry lake attributes
dryAreas, dryPerimeters, dryCounts = [],[],[]
# loop through ids and get dryer lake area, perimeter, count
for i in idList:
    print(i)
    df = drylakes2.loc[drylakes2['id']==i]
    dArea = sum(df['dryArea'])
    dPerimeter = sum(df['dryPerimeter'])
    dCount = len(df)
    dryAreas.append(dArea)
    dryPerimeters.append(dPerimeter)
    dryCounts.append(dCount)

# test list lenghts
assert len(dryAreas)==len(lakes)
assert len(dryPerimeters)==len(lakes)
assert len(dryCounts) == len(lakes)


# Add attributes to lake df
lakes['dryArea'] = dryAreas
lakes['dryPerimeter'] = dryPerimeters
lakes['dryCount'] = dryCounts
# get areas in km
lakes['areakm'] = lakes['area'] / 1000000
lakes['dryAreakm'] = lakes['dryArea'] / 1000000
# calculate change in area and perimeter
lakes['deltaArea'] = lakes['areakm'] - lakes['dryAreakm']
lakes['deltaPerim'] = lakes['perimeter'] - lakes['dryPerimeter']
## calculate delta Radius
lakes['deltaRad'] = (np.sqrt(lakes['areakm']/np.pi)) - (np.sqrt(lakes['dryAreakm']/np.pi))




# Export shapefile
try:
    lakes.to_file("/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/analysis/lakes.shp")
except Exception as e:
    print("Error during export:", e)
    
# Export csv 
lakes2 = lakes.drop(columns=['geometry'])
lakes2.to_csv("/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/analysis/lakes.csv")

    

#########################
#####STORING SHIT#####

# setting and fixing geometry
lakes = lakes.set_geometry(lakes['geometry'])
lakes['geometry'] = lakes['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
invalid_geoms = lakes[~lakes['geometry'].is_valid | lakes['geometry'].is_empty]
print(invalid_geoms)
lakes['centroid'] = lakes['centroid'].astype(str)
lakes['lat'] = 

print(lakes.iloc[0])
# Storage: Visualization
plt.scatter(np.log10(lakes['areakm']/(np.sqrt(lakes['areakm']/np.pi))), lakes['deltaRad'])
plt.xlim(np.log10(0),np.log10(1000))
plt.show()
np.min(lakes['deltaRad'])

plt.scatter(np.log10(lakes['areakm']), lakes['deltaRad'])
lakes25_2.columns
## Counting stats
len(lakes2)
len(drylakes)
lakes.columns
lakes['deltaArea'].sum() / 1723337 *100

lakes['areakm'].sum()/ 1723337 *100

lakes['dryAreakm'].sum()/ 1723337 *100
lakes['areakm'].std()
lakes['deltaRad'].min()





# Get the current tick locations and labels
tick_locations = plt.xticks()[0]
tick_labels = np.log10(lakes['areakm'])

# Convert tick locations to the corresponding log values
log_tick_locations = np.log10(tick_labels)

# Set the x-axis ticks and labels
plt.xticks(log_tick_locations, tick_labels)

# Display the plot
plt.show()
from matplotlib.ticker import FuncFormatter

# Plot the histogram with log-transformed data
plt.hist(np.log10(lakes['areakm']), bins='auto')
# Define a custom formatter function to convert log scale back to original scale
def log_to_original_scale(x, pos):
    return f"{int(10 ** x)}"
10 ** -2
# Set the custom formatter for the x-axis ticks
formatter = FuncFormatter(log_to_original_scale)
plt.gca().xaxis.set_major_formatter(formatter)
lakes.columns
lakes = lakes.loc[lakes['dryCount']==0]
len(lakes_eph)

plt.hist(problems['deltaRad'])
### looking into negative delta lakes
problems = lakes.loc[lakes['deltaRad']<0]
len(problems)
lakes_eph.to_file('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/analysis/lakes_eph.shp')
#coefficient of variation, which is std/mean

plt.hist(lakes['deltaRad'], bins='auto')
# coefficient of variation
lakes['areakm'].std()
lakes['cov'] = lakes

lakes['deltaRad'].median()*1000
### TODO: set negative deltaRads to zero or filter
lakes.loc[lakes['deltaRad']<0, 'deltaRad'] = 0


### Create lakePoints using centroid
lakePoints = lakes.drop(columns='centroid')
lakePoints['centroid'] = lakePoints['geometry'].centroid
lakePoints = lakePoints.drop(columns='geometry')
lakePoints = lakePoints.set_geometry('centroid')
lakePoints['centroid']
lakePoints = gpd.read_file('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/analysis/lakePointsPerm.shp')
lakePoints.columns
plt.scatter(lakePoints['PF_prob1'], lakePoints['deltaRad'], alpha=0.005)
plt.scatter(lakePoints['PF_prob1'], lakePoints['deltaArea']/lakePoints['areakm'], alpha=0.01)


###Combine lakePoints with pfro
pfro = gpd.read_file('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/GIS_layers/permafrost/UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH/UiO_PEX_PERZONES_5.0_20181128_2000_2016_NH.shp')
pfro = pfro.to_crs('EPSG:3338')
len(lakePoints)
lakePoints = gpd.sjoin(lakePoints, pfro, how='left')


import seaborn as sns
lakePoints['logArea'] = np.log10(lakePoints['areakm'])
lakePoints['logdeltaRad'] = np.log10(lakePoints['deltaRad'])
sns.scatterplot(x='logArea', y='logdeltaRad', data=lakePoints, hue='EXTENT', ec=None, alpha=0.1)

x = sns.scatterplot(x='logArea', y='logdeltaRad', data=lakePoints.loc[lakePoints['EXTENT']=='Cont'], ec=None, alpha=0.1)
x.set(xlim=(-3,2))
x.set(ylim=(-4,0))


plt.scatter(lakePoints['areakm'], lakePoints['deltaRad'])

sns.boxplot(data=lakePoints, x='EXTENT', y='dArea_norm')

lakePoints.columns

plt.hist(lakePoints['logdeltaRad'], bins=20,range=(-2.5,0))


g = sns.FacetGrid(lakePoints, col="EXTENT")
g.map(sns.scatterplot, "deltaArea", "deltaRad", alpha=0.4)

lakePoints['dArea_norm'] = (lakePoints['areakm']-lakePoints['deltaArea'])/lakePoints['areakm']
huh = lakePoints.loc[lakePoints['deltaRad']> 1.5]
len(huh)
huh.to_file('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/analysis/outliers.shp')

lakes = lakes.to_file('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/products/analysis/lakes2.shp')
