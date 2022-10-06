# Map Arctic lakes

### Part 1: LakeScan  |  Identify lakes while filtering out rivers, shadows, burn scars.

#### Scripts:
S2bgrnExport.ipynb OR S2maskedmosaic.ipynb
S2download.ipynb
S2imStack.ipynb
Predictor.ipynb
S2vectorize.ipynb (not yet committed)

### Part 2: LakeTrack  |  Identify precise locations of individual lakes and record their changes through time.

#### Scripts:
LakeTimeSeriesv1.ipynb
LakeTimeSeriesv2visualize.ipynb
TimeSeriesConcatv1.ipynb




This repository will eventually hold scripts to accomplish the following tasks: 
  (1) Creating an object-based water body mask (in development); 
  (2) Processing images in Earth Engine and exporting near-daily observations of water extent (v1 committed);
  (3) Time series analysis from the Earth Engine outputs (v1 committed)
