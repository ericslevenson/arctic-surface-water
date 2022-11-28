#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:44:14 2022

@author: elevens2

description: merge bufferd swot shapefile to unet

"""

# Imports

import geopandas as gpd
from geopandas import GeoDataFrame
import os
import pandas as pd
import numpy as np
#import pygeos
import rtree
 

## ****INPUTS****

# UTM zone
utmzone = '07'

# UNET buffered lake shapefile
UNETdirectory = '/Users/elevens2/Dropbox (University of Oregon)/ArcticLakeScan/buffered_UNET_output_vectors/07/'

# SWOT buffered lake shapefile
SWOTpath = '/Users/elevens2/Dropbox (University of Oregon)/ArcticLakeScan/GIS_layers/SWOT_lakes/SWOT_7N/SWOTLakes_7N_buffered.shp'

# Sentinel-2 Tile shapefile
S2tilePath = '/Users/elevens2/Dropbox (University of Oregon)/ArcticLakeScan/1mergeswot2unet/Sentinel-2-AK_Arctic_tiles/sentinel_2_AK_Arctic_tiles.shp'

# Export Location
exportFolder = '/Users/elevens2/Dropbox (University of Oregon)/ArcticLakeScan/buffered_lakes_uninspected/'



## ****METHODS***

def findID(unet_path):
  '''get the unique Sentinel-2 tile ID given a path. Returns a string '''
  utmz = str(utmzone)
  letterid = unet_path.split('_'+utmz)[1][:3]
  id = str(utmz + letterid)
  return id



## ****MAIN****

exportedIDs = [] # initialize list of exported tiles
S2tiles = gpd.read_file(S2tilePath) # read in s2tiles
swot = gpd.read_file(SWOTpath) # read in swot lakes

# create a list of files
files = []
for x in os.listdir(UNETdirectory):
    files.append(x)

# filter out files that have already been merged
files = [j for j in files if str(findID(j) + '_merged_unet_bufferedLakes.shp') not in os.listdir(exportFolder)]

# get total number of files
numberoffiles = str(len(files))
numberoffiles
for i, f in enumerate(files):
    if f.split('.')[1] == 'shp':
        print(str(i) + ': ' + f)
        
for i, f in enumerate(files):
    # select for shapefiles
    if f.split('.')[1] == 'shp':
        ID = findID(f) # get ID
        S2tile = S2tiles.loc[S2tiles['Name'] == ID] # select S2 tile
        S2tile = S2tile.to_crs(epsg = '326'+utmzone) # reproject S2tile
        # clip swot lakes to roi
        swot_clipped = gpd.clip(swot, S2tile)
        # read in unet buffered lakes
        unet = gpd.read_file(UNETdirectory+'/'+f)
        # merge clipped swot lakes
        unetswot = unet.append(swot_clipped)
        del unet
        del swot_clipped
        # Export 
        
        # check if ID exists for a different date
        if ID in exportedIDs:
            exportFileName = ID+'_merged_unet_bufferedLakes_2.shp'
        else:
            exportFileName = ID+'_merged_unet_bufferedLakes.shp'
            exportedIDs.append(ID)
        print('exporting ' + ID + ', # ' + str(i) + ' of ' + numberoffiles)
        # export to shapefile
        unetswot.to_file(exportFolder+ exportFileName) # export


