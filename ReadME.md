# Arctic Lake Database

Use Sentinel-2 to: (a) track variability in individual lakes' surface area, and (b) produce a lake occurrence mask across the Arctic.

#### Part 1: LakeScan   |   Scan the entire landscape to identify lakes while filtering out rivers, shadows, burn scars, etc.

#### Part 2: LakeTrack  |  Identify precise locations of individual lakes and record their changes through time.


### LakeScan Description
LakeScan tells us **where to point our water classification** and also provides the basis for **tracking lakes as objects** rather than unrelated pixels. The output of LakeScan is a shapefile of buffered lakes, which is in turn the input for LakeTrack.

##### LakeScan Scripts:
S2bgrnExport.ipynb OR S2maskedmosaic.ipynb \
      *Rapidly filter, mask, visualize, and export Sentinel-2 images from GEE.* \
S2download.ipynb \
    *Batch download Sentinel-2 images* \
S2imStack.ipynb \
    *Combine the R,G,B,NIR Sentinel-2 downloads into 4-band images* \
Predictor.ipynb \
    *Use the UNET model to predict lake locations* \
dilate2vectorize.ipynb \
    *Buffer UNET lake predictions and convert to shapefile* \
merge_swot2unet.py \
    *merge buffered SWOT lake shapefile to UNET outputs*
  
### LakeTrack Description:
Based on all available Sentinel-2 images, LakeTrack provides a final **lake occurrence mask** for customizable areas and timeframes as well as a **timeseries of each lake's surface area**. \

##### LakeTrack Scripts:
LakeOccurrence.ipynb \
     *Process images in Earth Engine and export a lake occurrence composite image* \
LakeOccurrenceBatches.ipynb \
    *Batch processing of ROIs to process images in Earth Engine and export a lake occurrence composite* \
OccurrenceProcessing.py \
   *Clean lake occurrence rasters using segmentation techniques, and convert to shapefiles* \
mask2seed.ipynb \
      *Create new buffered lake shapefile based on the lake occurrence raster* \
LakeTimeSeriesv3.ipynb \
    *Process images in Earth Engine and export near-daily observations of water extent per lake* \
LakeTimeSeriesv1.ipynb \
    *Process images in Earth Engine and export near-daily observations of water extent* \
LakeTimeSeriesv2visualize.ipynb \
    *Process images in Earth Engine and export a lake occurrence composite image* \
TimeSeriesConcatv1.ipynb \
    *Combine Earth Engine exports into indexed timeseries for each lake.*

