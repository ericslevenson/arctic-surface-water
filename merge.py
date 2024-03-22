#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:58:49 2023

@author: ericlevenson

Description: Merge rasters by utm zone
"""

## ***IMPORTS***
import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.enums import Resampling

## ***INPUTS***
utmzone = '7N'
thresholds = '10'
user = 'ericlevenson' # ericlevenson = laptop; elevens2 = desktop
in_directory = f'/Users/{user}/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/zeros/' # Path to lakeOccurrence Folder
out_directory = f'/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/binaries/'
output_filename = f'{utmzone}_LakeOccurrence_{thresholds}.tif'
in_directory = f'/Users/{user}/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/binaries/'
out_directory = f'/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/rasters/'
output_filename = f'{utmzone}_LakeOccurrence_{thresholds}_binary.tif'
# for merging binaries
#in_directory = '/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/binaries/fifty/'
#out_path = '/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/binaries/fifty/AK_50_binary.tif'

## ***METHODS***

# Function to merge rasterio data keeping maximum
def custom_merge(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    old_data[:] = np.maximum(old_data, new_data)  # <== NOTE old_data[:] updates the old data array *in place*


# Get a list of file paths
input_files = []
for i in os.listdir(in_directory):
    path = str(in_directory + i)
    input_files.append(path)
    
    # Get a list of rasterio files
datasets = []
for file in input_files:
    if not file.endswith('.DS_Store'):
        datasets.append(rasterio.open(file))
            
# Calculate the bounding box of the merged raster
merged_bounds = datasets[0].bounds
for ds in datasets[1:]:
    merged_bounds = (
        min(merged_bounds[0], ds.bounds[0]),
        min(merged_bounds[1], ds.bounds[1]),
        max(merged_bounds[2], ds.bounds[2]),
        max(merged_bounds[3], ds.bounds[3])
    )
        
# Create mosaic    
mosaic, out_transform = merge(datasets, bounds=merged_bounds, method = custom_merge)
mosaic = mosaic[0]
# Update metadata
out_meta = datasets[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[0],
    "width": mosaic.shape[1],
    "transform": out_transform,
    "compress":'lzw',
    'nodata':0
})
del datasets    
#Write raster
with rasterio.open(str(out_directory+output_filename), 'w', **out_meta) as dst:
    dst.write(mosaic, 1)

## work to binarize mosaic
binary = np.where(mosaic < 25, 0, 1).astype('int16')

with rasterio.open(str(out_directory+f'{utmzone}_LakeOccurrence_{thresholds}_ns_binary.tif'), 'w', **out_meta) as dst:
    dst.write(binary, 1)



#####################
## for opening and binarizing an image

with rasterio.open('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/South/rasters/South_LakeOccurrence_50b.tif') as src:
    im = src.read(1)
    meta = src.meta
    meta.update(compress = 'lzw')
binary = np.where(im < 13, 0, 1).astype('int16')
with rasterio.open('/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/South/rasters/South_LakeOccurrence_75a_binary.tif', 'w', **meta) as dst:
    dst.write(binary, 1)
    
    
## binarizing a bunch of images
for i in input_files:
    if i.endswith('.tif'):
        with rasterio.open(i) as src:
           im = src.read(1)
           meta = src.meta
           meta.update(compress = 'lzw')
           out_des = str(i.split('zeros/')[1].split('_')[0]+'_10_binary.tif')
           print(f'binarizing {out_des}')
           binary = np.where(im < 13, 0, 1).astype('int16')
           print(f'Writing {out_des}')
           with rasterio.open(str(out_directory+out_des), 'w', **meta) as dst:
               dst.write(binary, 1)
    else:
        continue
        
str(i.split('zeros/')[1].split('_')[0]+'_10_binary.tif')
i
