#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:23:48 2023

@author: ericlevenson

description: Clean up Lake Occurrence Rasters with a mask based on segmentation
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


## ***INPUTS***
computer = 'laptop' #laptop or desktop
mode = 'batch'  ### single or batch for one image or entire folder
directory = '/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeTrack/lakeOccurrence/5N/unprocessed/' # Path to lakeOccurrence Folder
tile_id =  199 # use in single mode
out_directory = '/Users/ericlevenson/Dropbox (University of Oregon)/ArcticLakeTrack/lakeOccurrence/5N/processed/' # Path to processed lake Occurrence Folder
input_descriptor = '_5N_lakeOccurrence_2016-2021.tif'
output_descriptor = '_5N_lakeOccurrence_2016-2021_processed.tif'


## ***Methods***

def im_process(image):
    '''Perform image processing on numpy array input'''
    # Edge detection
    edges = sobel(image)
    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < 0.25] = background
    markers[image > 0.3] = foreground
    # Watershed Segmentation
    ws = watershed(edges, markers)
    # Label Segments
    seg1 = label(ws == foreground)
    # Dilate Labels
    expanded = expand_labels(seg1, distance=6)
    # Convert 'expanded' to a boolean mask
    expanded_mask = expanded.astype(bool)
    # Apply the boolean mask to the 'image' variable
    image[~expanded_mask] = np.nan
    return image

def visualize_processing(image):
    ''''Process image and produce plot for intermediate steps'''
    # Edge detection
    edges = sobel(image)
    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < 0.25] = background
    markers[image > 0.3] = foreground
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
        impath = str(directory + str(tile_id) + '_5N_lakeOccurrence_2016-2021.tif')           
        # open image and record metadata
        with rasterio.open(impath, dtype='float64') as dataset:
            # Access the image data
            image = dataset.read(1)  # Assuming you want to read the first band
            #image[np.isnan(image)] = 0.0
            meta = dataset.meta
            meta.update(compress='lzw') # reduce file size
            # Segment image
            image = im_process(image)
            # Write the modified 'image' variable to a new raster file
            export_descriptor = str(str(tile_id) + output_descriptor)
            with rasterio.open(str(out_directory + export_descriptor), 'w', **meta) as dst:
                dst.write(image, 1)
    elif mode == 'batch':
        processed = [i.split('_')[0] for i in os.listdir(out_directory) if output_descriptor in i] # list of processed tiles
        images = [i for i in os.listdir(directory) if input_descriptor in i and i.split('_')[0] not in processed] # get list of images
        print(f'Hi Eric! Gonna process {len(images)} images. There are {len(processed)} previously processed images.')
        for x, i in enumerate(images):
            impath = str(directory+i)
            print(f'Writing image {x+1} of {len(images)}.')
            with rasterio.open(impath, dtype='float64') as dataset:
                # Access the image data
                image = dataset.read(1)  # Assuming you want to read the first band
                #image[np.isnan(image)] = 0.0
                meta = dataset.meta
                meta.update(compress='lzw')
                # Segment image
                image = im_process(image)
                # Write the modified 'image' variable to a new raster file
                export_descriptor = str(str(i.split('_')[0] + output_descriptor))
                with rasterio.open(str(out_directory + export_descriptor), 'w', **meta) as dst:
                    dst.write(image, 1)
        print('Well, that was fun!')
    else:
        print('ERROR: Set mode input variable to single or batch')
        