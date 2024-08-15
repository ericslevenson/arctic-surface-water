#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:44:52 2024

@author: ericlevenson
"""
import ee
import logging
import multiprocessing
from retry import retry


ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

start = '2016-03-01'
finish = '2023-10-01'

## Periodically Adjusted Inputs
# Lake Shapefiles
lakes = ee.FeatureCollection('projects/ee-eric-levenson/assets/PLD/PLD_60n0e_buff')
lakes = lakes.merge(ee.FeatureCollection('projects/ee-eric-levenson/assets/PLD/PLD_60n110w_buff'))
lakes = lakes.merge(ee.FeatureCollection('projects/ee-eric-levenson/assets/PLD/PLD_60n180w_buff'))
lakes = lakes.merge(ee.FeatureCollection('projects/ee-eric-levenson/assets/PLD/PLD_60n60w_buff'))
lakes = lakes.merge(ee.FeatureCollection('projects/ee-eric-levenson/assets/PLD/PLD_60n90e_buff'))

#Basins

# exports
#exportSelectors = ['id', 'date', 'waterArea', 'coverage']



  
  ## ***MAIN***
@retry(tries=10, delay=1, backoff=2)
def getResult(index, basinID):
    #(1) define roi
    ## Rarely Adjusted Inputs ##
    # Image scale
    pixScale = 10
    # Cloud probability threshold
    CLD_PRB_THRESH = 50
    ## ***EARTH ENGINE-IFY***
    startDoy = ee.Date(start).getRelative('day', 'year')
    endDoy = ee.Date(finish).getRelative('day', 'year')
    eestart = ee.Date(start)

    basin_l10 = ee.FeatureCollection("projects/sat-io/open-datasets/HydroAtlas/BasinAtlas/BasinATLAS_v10_lev10")

    ###############################################################################
    ## ***IMAGE PRE-PROCESSING METHODS***

    # Mask clouds in Sentinel-2
    def add_cloud_bands(img):
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

        # Condition s2cloudless by the probability threshold value.
        clouds = cld_prb.gte(CLD_PRB_THRESH).rename('cloud_mask')
        clear = cld_prb.lt(CLD_PRB_THRESH).rename('clear_mask')
        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, clouds, clear]))

    # Clip image
    def clip_image(image):
      '''Clips to the roi defined at the beginning of the script'''
      return image.clip(roi)

    def clip2lakes(image):
      '''Clips an image based on the lake boundaries'''
      return image.clip(lakes)

    # Get percentile cover
    def getCover(image):
      '''calculates percentage of the roi covered by the clear mask. NOTE: this function
      calls the global totPixels variable that needs to be calculated in the main script.'''
      actPixels = ee.Number(image.updateMask(image.select('clear_mask')).reduceRegion(
          reducer = ee.Reducer.count(),
          scale = 100, # keep same as totPixels
          geometry = image.geometry(),
          maxPixels=1e12,
          ).values().get(0))
      # calculate the perc of cover OF CLEAR PIXELS
      percCover = actPixels.divide(totPixels).multiply(100).round()
      # number as output
      return image.set('percCover', percCover,'actPixels',actPixels)

    # Mosaic images by date, orbit, - basically combines images together that were taken on the same day
    def mosaicBy(imcol):
      '''Takes an image collection (imcol) and creates a mosaic for each day
      Returns: An image collection of daily mosaics'''
      #return the collection as a list of images (not an image collection)
      imlist = imcol.toList(imcol.size())
      # Get all the dates as list
      def imdate(im):
        date = ee.Image(im).date().format("YYYY-MM-dd")
        return date
      all_dates = imlist.map(imdate)
      # get all orbits as list
      def orbitId(im):
        orb = ee.Image(im).get('SENSING_ORBIT_NUMBER')
        return orb
      all_orbits = imlist.map(orbitId)
      # get all spacecraft names as list
      def spacecraft(im):
        return ee.Image(im).get('SPACECRAFT_NAME')
      all_spNames = imlist.map(spacecraft)
      # this puts dates, orbits and names into a nested list
      concat_all = all_dates.zip(all_orbits).zip(all_spNames);
      # here we unnest the list with flatten, and then concatenate the list elements with " "
      def concat(el):
        return ee.List(el).flatten().join(" ")
      concat_all = concat_all.map(concat)
      # here, just get distinct combintations of date, orbit and name
      concat_unique = concat_all.distinct()
      # mosaic
      def mosaicIms(d):
        d1 = ee.String(d).split(" ")
        date1 = ee.Date(d1.get(0))
        orbit = ee.Number.parse(d1.get(1)).toInt()
        spName = ee.String(d1.get(2))
        im = imcol.filterDate(date1, date1.advance(1, "day")).filterMetadata('SPACECRAFT_NAME', 'equals', spName).filterMetadata('SENSING_ORBIT_NUMBER','equals', orbit).mosaic()
        return im.set(
            "system:time_start", date1.millis(),
            "system:date", date1.format("YYYY-MM-dd"),
            "system:id", d1)
      mosaic_imlist = concat_unique.map(mosaicIms)
      return ee.ImageCollection(mosaic_imlist)

    ###########################################################################
    ## ***WATER CLASSIFICATION METHODS***

    # Define NDWI image
    def ndwi(image):
      '''Adds an NDWI band to the input image'''
      return image.normalizedDifference(['B3', 'B8']).rename('NDWI').multiply(1000)

    # Basic ndwi classification
    def ndwi_classify(image):
      '''Creates a binary image based on an NDWI threshold of 0'''
      ndwimask = image.select('NDWI')
      water = ndwimask.gte(0)
      land = ndwimask.lt(0)
      return(water)

    # OTSU thresholding from histogram
    def otsu(histogram):
      '''Returns the NDWI threshold for binary water classification'''
      counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
      means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
      size = means.length().get([0])
      total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
      sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
      mean = sum.divide(total)
      indices = ee.List.sequence(1, size)
      def func_xxx(i):
        '''Compute between sum of squares, where each mean partitions the data.'''
        aCounts = counts.slice(0, 0, i)
        aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
        aMeans = means.slice(0, 0, i)
        aMean = aMeans.multiply(aCounts) \
            .reduce(ee.Reducer.sum(), [0]).get([0]) \
            .divide(aCount)
        bCount = total.subtract(aCount)
        bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
        return aCount.multiply(aMean.subtract(mean).pow(2)).add(
               bCount.multiply(bMean.subtract(mean).pow(2)))
      bss = indices.map(func_xxx)
      # Return the mean value corresponding to the maximum BSS.
      return means.sort(bss).get([-1])

    # OTSU thresholding for an image
    def otsu_thresh(water_image):
      '''Calculate NDWI and create histogram. Return the OTSU threshold.'''
      NDWI = ndwi(water_image).select('NDWI').updateMask(water_image.select('clear_mask'))
      histogram = ee.Dictionary(NDWI.reduceRegion(
        geometry = roi,
        reducer = ee.Reducer.histogram(255, 2).combine('mean', None, True).combine('variance', None, True),
        scale = pixScale,
        maxPixels = 1e12
      ))
      return otsu(histogram.get('NDWI_histogram'))

    # Classify an image using OTSU threshold.
    def otsu_classify(water_image):
      '''(1) Calculate NDWI and create histogram. (2) Calculate NDWI threshold for
      binary classification using OTSU method. (3) Classify image and add layer to input image.
      '''
      NDWI = ndwi(water_image).select('NDWI')
      histogram = ee.Dictionary(NDWI.reduceRegion(
        geometry = roi,
        reducer = ee.Reducer.histogram(255, 2).combine('mean', None, True).combine('variance', None, True),
        scale = pixScale,
        maxPixels = 1e12
      ))
      threshold = otsu(histogram.get('NDWI_histogram'))
      otsu_classed = NDWI.gt(ee.Number(threshold)).And(water_image.select('B8').lt(2000)).rename('otsu_classed')
      return water_image.addBands([otsu_classed])

    def adaptive_thresholding(water_image):
      '''Takes an image clipped to lakes and returns the water mask'''
      NDWI = ndwi(water_image).select('NDWI').updateMask(water_image.select('clear_mask')) # get NDWI **TURNED OFF CLOUD MASK, because i do it in the mains cript
      threshold = ee.Number(otsu_thresh(water_image))
      threshold = threshold.divide(10).round().multiply(10)
      # get fixed histogram
      histo = NDWI.reduceRegion(
          geometry = roi,
          reducer = ee.Reducer.fixedHistogram(-1000, 1000, 200),
          scale = pixScale, # This was 30, keep at 10!?!?
          maxPixels = 1e12
      )
      hist = ee.Array(histo.get('NDWI'))
      counts = hist.cut([-1,1])
      buckets = hist.cut([-1,0])
      #find split points from otsu threshold
      threshold = ee.Array([threshold]).toList()
      buckets_list = buckets.toList()
      split = buckets_list.indexOf(threshold)
      # split into land and water slices
      land_slice = counts.slice(0,0,split)
      water_slice = counts.slice(0,split.add(1),-1)
      # find max of land and water slices
      land_max = land_slice.reduce(ee.Reducer.max(),[0])
      water_max = water_slice.reduce(ee.Reducer.max(),[0])
      land_max = land_max.toList().get(0)
      water_max = water_max.toList().get(0)
      land_max = ee.List(land_max).getNumber(0)
      water_max = ee.List(water_max).getNumber(0)
      #find difference between land, water and otsu val
      counts_list = counts.toList()
      otsu_val = ee.Number(counts_list.get(split))
      otsu_val = ee.List(otsu_val).getNumber(0)
      land_prom = ee.Number(land_max).subtract(otsu_val)
      water_prom = ee.Number(water_max).subtract(otsu_val)
      #find land and water buckets corresponding to 0.9 times the prominence
      land_thresh = ee.Number(land_max).subtract((land_prom).multiply(ee.Number(0.9)))
      water_thresh = ee.Number(water_max).subtract((water_prom).multiply(ee.Number(0.9)))
      land_max_ind = land_slice.argmax().get(0)
      water_max_ind = water_slice.argmax().get(0)
      li = ee.Number(land_max_ind).subtract(1)
      li = li.max(ee.Number(1))
      wi = ee.Number(water_max_ind).add(1)
      wi = wi.min(ee.Number(199))
      land_slice2 = land_slice.slice(0,li,-1).subtract(land_thresh)
      water_slice2 = water_slice.slice(0,0,wi).subtract(water_thresh)
      land_slice2  = land_slice2.abs().multiply(-1)
      water_slice2 = water_slice2.abs().multiply(-1)
      land_index = ee.Number(land_slice2.argmax().get(0)).add(land_max_ind)
      water_index = ee.Number(water_slice2.argmax().get(0)).add(split)
      land_level = ee.Number(buckets_list.get(land_index))
      water_level = ee.Number(buckets_list.get(water_index))
      land_level = ee.Number(ee.List(land_level).get(0)).add(5)
      water_level = ee.Number(ee.List(water_level).get(0)).add(5)
      #calculate water fraction and classify
      water_fraction = (NDWI.subtract(land_level)).divide(water_level.subtract(land_level)).multiply(100).rename('water_fraction')
      #water_fraction = conditional(water_fraction) #sets values less than 0 to 0 and greater than 100 to 100
      water_75 = water_fraction.gte(75).rename('water_75'); #note, this is a non-binary classification, so we use 75% water as "water"
      #all_mask = water_image.select('B2').gt(5).rename('all_mask')
      cloud_mask_ed = water_image.select('cloud_mask').rename('cloud_mask_ed')
      return water_image.addBands([water_fraction,water_75,NDWI,cloud_mask_ed])

    def binaryImage(image):
      '''takes a multiband image and returns just the binary water_75 band'''
      img = image.select('water_75')
      return img
    def waterImage(image):
      '''takes a multiband image and returns just the water fraction band'''
      img = image.select('water_fraction')
      return img


    def pixelArea(image):
      areaIm = image.pixelArea()
      lakeIm = image.addBands([areaIm])
      return lakeIm

    def sumWater(image):
      '''sums the water pixels within the watershed image and adds the result to the feature'''
      waterAreaIm = image.updateMask(image.select('water_75')) # mask area image based on water
      # calculate the total area
      watersum = waterAreaIm.select('area').reduceRegion(
          reducer=ee.Reducer.sum(),
          geometry = image.geometry(),
          scale = 10,
          maxPixels=1e9
      ).get('area')
      return watersum

    def sumIce(image):
      '''sums the ice pixels within the watershed image and adds the result to the feature'''
      iceAreaIm = image.updateMask(image.select('ice')) # mask area image based on ice
      # calculate the total area
      icesum = iceAreaIm.select('area').reduceRegion(
          reducer=ee.Reducer.sum(),
          geometry = image.geometry(),
          scale = 100,
          maxPixels=1e9
      ).get('area')
      return icesum

    def getClearArea(image):
      '''sums the clear pixels within the watershed image and adds the result to the feature'''
      clearAreaIm = image.updateMask(image.select('clear_mask')) # mask area image based on clearness
      clearArea = clearAreaIm.select('area').reduceRegion(
          reducer=ee.Reducer.sum(),
          geometry = image.geometry(),
          scale = 10,
          maxPixels=1e9
      ).get('area')
      return clearArea

    def getCloudArea(image):
      '''sums the cloud pixels within the watershed image and adds the result to the feature'''
      cloudAreaIm = image.updateMask(image.select('cloud_mask')) # mask area image based on clouds
      cloudArea = cloudAreaIm.select('area').reduceRegion(
          reducer=ee.Reducer.sum(),
          geometry = image.geometry(),
          scale = 10, ## TODO: I updated this scale from 10 to 100, does that change results?
          maxPixels=1e9
      ).get('area')
      return cloudArea

    def dayProps(image):
      '''Input: classified water Image --> Output: Feature with properties such as date, water area, clear area, etc.
      Map this function to the image collection of classified lake area images. This function calls
      methods to calculate the water area, clear area, and cloud area. '''
      #basin = id #TODO: FIXME
      date = image.date().format('yyyy-MM-dd')
      water = sumWater(image)
      #ice = sumIce(image)
      cover = image.get('percCover')
      #iceCover = image.get('percIceCover')
      #clear = getClearArea(image)
      #cloud = getCloudArea(image)
      return image.set('output', [date, water, cover])

    basin = ee.Feature(basin_l10.filter(ee.Filter.eq('HYBAS_ID', basinID)).first())
    roi = ee.Geometry.Polygon(basin.geometry().getInfo()['coordinates'][0]) # define roi as geometry variable

    # (2) image pre-process:
    # Get images and filter images by cloudiness, roi, time period, and month range
    images = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(roi).filterDate(start,finish).filter(ee.Filter.calendarRange(startDoy, endDoy, 'day_of_year')).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',60)) # Get Images
    # Get cloud probability image collection
    s2Cloudless = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(roi).filterDate(start,finish).filter(ee.Filter.calendarRange(startDoy, endDoy, 'day_of_year'))
    # Merge surface reflectance and cloud probability collections
    images = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': images,
            'secondary': s2Cloudless,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        }))
    images = images.map(add_cloud_bands)
    # Mosaic images and add cloud/clear masks
    images_all = mosaicBy(images)
    # Clip mosaics to roi
    images_all = images_all.map(clip_image)
    # Clip remaining mosaics to buffered lake shapefile
    lakeimages = images_all.map(clip2lakes) # Clip images to buffered lake mask

    # Get percent cover for each mosaic
    image_mask = lakeimages.select('B2').mean().gte(0) #
    # Calculate total number of pixels
    totPixels = ee.Number(image_mask.reduceRegion(
        reducer = ee.Reducer.count(),
        scale = 100,
        geometry = roi,
        maxPixels = 1e12
        ).values().get(0))
    lakeimages = lakeimages.map(getCover) # add percentage cover as an image property
    # Filter by percent cover
    lakeimages = lakeimages.filterMetadata('percCover','greater_than',70) # remove images covering less than 70% of the ROI)
    # (3) Classify water
    lakeimages = lakeimages.map(adaptive_thresholding)
  #  lakeimages = lakeimages.map(ice_classify)
    # Convert to area images
    lakeimages = lakeimages.map(pixelArea) # get a pixel area image
    #(4) Calculate water and ice area, convert to table format
    lakeimages = lakeimages.map(dayProps)
    result = lakeimages.aggregate_array('output').getInfo()
    # Write file
        
    #(5) Export
    filename = str(basinID)+'.csv'
    directory = '/Users/ericlevenson/Library/CloudStorage/OneDrive-UniversityOfOregon/arcticPulse/TS/'
    with open(directory+filename, 'w') as out_file:
        for items in result:
            line = ','.join([str(item) for item in items])
            print(line, file=out_file)
    print("Done: ", index, flush=True)

  


if __name__ == '__main__':
  logging.basicConfig()
  # Tile of Interest ID
  ids=[8100147170,8100151680, 8100384290]
  pool = multiprocessing.Pool(25)
  pool.starmap(getResult, enumerate(ids))
  pool.close()
  pool.join()