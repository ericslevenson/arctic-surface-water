#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:20:16 2023

@author: ericlevenson

description: Create copies of lake occurrence replacing nan's with zeros. This is important to merge adjacent rasters because of a numpy quirk that takes nans over floats in its max function.
"""

#TODO: Script to merge the tiles into utm zones of lakes
import numpy as np
import os
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.enums import Resampling

## ***INPUTS***
thresholds = '5-10'
utmzone = '7N'
user = 'ericlevenson' # ericlevenson = laptop; elevens2 = desktop
directory = f'/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/rasters_{thresholds}_new/' # Path to lakeOccurrence Folder
raster_out_directory = f'/Users/{user}/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/zeros/' # Path to processed lake Occurrence Folder
input_descriptor = f'_lakeOccurrence_2016-2021_processed_{thresholds}.tif'
output_descriptor = f'_{utmzone}_lakeOccurrence_2016-2021_{thresholds}_zeros.tif'



## ***MAIN***


if __name__ == "__main__":
    for i in os.listdir(directory):
        print(i)
        if i.endswith('.tif'):
            path = str(directory + i)
            with rasterio.open(path) as src:
                meta = src.meta
                data = src.read(1)
                data[np.isnan(data)] = 0.0
                data = data * 255
                meta.update(dtype='int16')
                binary = np.where(data < 13, 0, 1).astype('int16')
                meta.update(compress='lzw')
                tile = str(i.split('_')[0])
                export_descriptor = str(tile + output_descriptor)
                with rasterio.open(str(raster_out_directory + export_descriptor), 'w', **meta) as dst:
                        dst.write(binary, 1)