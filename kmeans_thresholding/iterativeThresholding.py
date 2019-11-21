import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from auxiliary_functions.contourFunctions import  *
from auxiliary_functions.thresholdingFunctions import *
from kmeans_thresholding.segmentedThresholding import *
from shape_information.Shape import *
from image_processing.image_filters import *
from image_processing.cropping.cropImage import *

def iterativeThresholding(image, fileName, rowsList, columnsList, kthreshList, filterList, pixelThreshList, uniList, factorList, save = False):
	'''
	Iteratively thesholds provided image
	Returns binary image, image with contours overlaid, and list of contours
	Arguments:
		image : pre-processed image (re-scaled and cropped)
		fileName : image filename
		rowsList : list of rows for segmentedThresholding call in each iteration i.e. [1,1,1]
		columnsList : list of columns for segmentedThresholding call in each iteration i.e. [3,3,3]
		kthreshList : list of (increasing) kthresh for segmentedThresholding call in each iteration i.e. [5,10,20]
		filterList : list of filters for segmentedThresholding call in each iteration i.e. [None, None, None] or [[gaussianFilter],[histogramEqualization],[histogramEqualization, gaussianFilter]]
		pixelThreshList : list of pixelThresh booleans for segmentedThresholding call in each iteration i.e. [False,False,False]
		uniList : list of uni for segmentedThresholding call in each iteration i.e. [True,True,True]
		factorList : list of factors for segmentedThresholding call in each iteration i.e. [0.3,0.3,0.3]
		save = False
	'''
	original_image = image.copy()
	for i in range(len(kthreshList)):
		binaryImage, contourImage, contours = segmentedThresholding(image,\
			rowsList[i],\
			columnsList[i],\
			kthreshList[i],\
			fileName,\
			filter = filterList[i],\
			save = save,\
			pixelThresh = pixelThreshList[i],\
			uni = uniList[i],\
			factor = factorList[i],\
			excludePixels = 250)

		#Save the initial binary image for watershed seeds
		if not(i):
			seeds = binaryImage

		image[binaryImage == 0] = 250

	binaryImage[image == 250] = 0
	contourImage, contours = drawShapes(binaryImage, original_image)
	return original_image, binaryImage, contourImage, contours, seeds



"""
#Example Run

#get image, crop, and rescale
image = cv2.imread('../test_images/120.tif',-1)
img = cropImage(image, cropTop = 100)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img = np.array(img_rescaled, dtype = np.uint8)

originalImage, binaryImage, contourImage, contours, seeds = iterativeThresholding(img,\
	'120.tif',\
	[1,1,1],\
	[1,1,1],\
	[5, 10, 20],\
	[None, None, None],\
	[False, False, False],\
	[True, True, True],\
	[1, 1, 1],\
	save = False)
for contour in contours:
	shape = Shape(contour, originalImage)
	shape.dropDivide(seeds)
plt.imshow(contourImage, cmap = 'gray')
plt.show()
"""
