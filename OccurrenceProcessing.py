#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:59:40 2023

@author: ericlevenson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:23:48 2023

@author: ericlevenson

description: Re-trying lake occurrence sans buffer
    
Clean up Lake Occurrence Rasters with a mask. based on segmentation
techniques that separate the foreground from background. Dilating the 
segmented image produces a useful mask to filter the original image.
"""


# Imports
import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import expand_labels, watershed
from geopandas import GeoDataFrame
from pandas import DataFrame
from rasterio.features import shapes
from shapely.geometry import shape


## ***INPUTS***
user = 'ericlevenson' # ericlevenson = laptop; elevens2 = desktop
utmzone = 'south'
vectors = False
mode = 'batch'  ### single or batch for one image or entire folder
background_threshold = 10 # low end threshold
foreground_threshold = 25 # markers for watershed segmentation
directory = f'/Users/{user}/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/unprocessed_originals/' # Path to lakeOccurrence Folder
tile_id =  342 # use in single mode
raster_out_directory = f'/Users/{user}/Dropbox (University of Oregon)/ArcticLakeScan/postProcessing/{utmzone}/rasters_5-10/' # Path to processed lake Occurrence Folder
shapefile_out_directory = '/Users/{user}/Dropbox (University of Oregon)/ArcticLakeScan/timeSeries/vectors/'
input_descriptor = f'.tif'
output_descriptor = f'_processed_{str(background_threshold)}-{str(foreground_threshold)}.tif'


## ***Methods***
def im_process(image, background_threshold, foreground_threshold):
    '''Perform image processing on numpy array input'''
    # Edge detection
    #edges = sobel(image)
    image[np.isnan(image)] = 0.0
    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < background_threshold] = background
    markers[image > foreground_threshold] = foreground
    # Watershed Segmentation
    ws = watershed(image, markers)
    # Label Segments
    seg1 = label(ws == foreground)
    # Convert seg1 to a boolean mask
    boolmask = seg1.astype(bool)
    # Apply the boolean mask to the 'image' variable
    image[~boolmask] = np.nan
    # Binarize segmented and expanded
    seg1 = np.where(seg1 == 0, 0, 1).astype(np.int32)
    #expanded = np.where(expanded == 0, 0, 1).astype(np.int32)
    return image, seg1

def vectorize(segmented):
    '''Produces geodataframes for a segmented and expanded segmentation raster'''
    shape_gen = ((shape(s), v) for s, v in shapes(segmented, transform=dataset.transform))
    # build a pd.DataFrame of shapes and convert to geodataframe
    df_seg = DataFrame(shape_gen, columns=['geometry', 'class'])
    gdf_seg = GeoDataFrame(df_seg["class"], geometry=df_seg.geometry, crs=dataset.crs)
    # filter out non-lake polygons
    gdf_seg = gdf_seg[gdf_seg['class'] == 1]
    del shape_gen
    #shape_gen = ((shape(s), v) for s, v in shapes(expanded, transform=dataset.transform))
    #df_expand = DataFrame(shape_gen, columns=['geometry', 'class'])
    #gdf_expand = GeoDataFrame(df_expand["class"], geometry=df_expand.geometry, crs=dataset.crs)
    # filter out non-lake polygons
    #gdf_expand = gdf_expand[gdf_expand['class'] == 1]
    return gdf_seg
    
def visualize_processing(image, background_threshold, foreground_threshold):
    ''''Process image and produce plot for intermediate steps'''
    # Edge detection
    edges = sobel(image)
    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < background_threshold] = background
    markers[image > foreground_threshold] = foreground
    # Watershed Segmentation
    ws = watershed(edges, markers)
    # Label Segments
    seg1 = label(ws == foreground)
    # Dilate Labels
    expanded = expand_labels(seg1, distance=6)
    # Show the segmentations.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )
    axes[0].imshow(image, cmap="Greys_r")
    axes[0].set_title("Original")
    color1 = label2rgb(seg1, image=image, bg_label=0)
    axes[1].imshow(color1)
    axes[1].set_title("Sobel+Watershed")
    color2 = label2rgb(expanded, image=image, bg_label=0)
    axes[2].imshow(color2)
    axes[2].set_title("Expanded labels")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if mode == 'single':
        # Create image path
        impath = str(directory + str(tile_id) + '_{utmzone}_lakeOccurrence_2016-2021.tif')           
        # open image and record metadata
        with rasterio.open(impath, dtype='float64') as dataset:
            # Access the image data
            image = dataset.read(1)  # Assuming you want to read the first band
            #image[np.isnan(image)] = 0.0
            meta = dataset.meta
            meta.update(compress='lzw') # reduce file size
            # Segment image
            image, segmented = im_process(image, background_threshold/100, foreground_threshold/100)
            # Write the modified 'image' variable to a new raster file
            export_descriptor = str(str(tile_id) + output_descriptor)
            with rasterio.open(str(raster_out_directory + export_descriptor), 'w', **meta) as dst:
                dst.write(image, 1)
            # Vectorize
            gdf_seg = vectorize(segmented)
            # Write the lake shapefile
            gdf_seg.to_file(shapefile_out_directory + str(tile_id) + '.shp')
            # Write the buffered lake shapefile
            #gdf_expand.to_file(buffered_shapefile_out_directory + str(tile_id) + '_buffered.shp')
    elif mode == 'batch':
        processed = [str(i.split('_')[0] + '_' + i.split('_')[1].split('.')[0]) for i in os.listdir(raster_out_directory) if output_descriptor in i] # list of processed tiles
        images = [i for i in os.listdir(directory) if input_descriptor in i and str(i.split('_')[0] + '_' + i.split('_')[2].split('.')[0]) not in processed] # get list of images
        print(f'Hi! Gonna process {len(images)} images. There are {len(processed)} previously processed images.')
        for x, i in enumerate(images):
            impath = str(directory+i)
            print(f'Writing image {i}, which is {x+1} of {len(images)}.')
            with rasterio.open(impath, dtype='float64') as dataset:
                # Access the image data
                image = dataset.read(1)  # Assuming you want to read the first band
                #image[np.isnan(image)] = 0.0
                meta = dataset.meta
                meta.update(compress='lzw')
                # Segment image
                image, segmented = im_process(image, background_threshold/100, foreground_threshold/100)
                # Write the modified 'image' variable to a new raster file
                export_descriptor = str(str(i.split('_')[0] + '_' + str(i.split('_')[2].split('.')[0]) + output_descriptor))
                
                with rasterio.open(str(raster_out_directory + export_descriptor), 'w', **meta) as dst:
                    dst.write(image, 1)
            # Vectorize
            if vectors == True:
                gdf_seg = vectorize(segmented)
                # Write the lake shapefile
                gdf_seg.to_file(shapefile_out_directory + str(i.split('_')[0]) + '_' + str(i.split('_')[2].split('.')[0])+'_'+str(foreground_threshold) + '.shp')
            else:
                continue
        print('Well, that was fun!')
    else:
        print('ERROR: Set mode input variable to single or batch')
    