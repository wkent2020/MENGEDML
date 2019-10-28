import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
if '../auxiliary_functions/' not in sys.path:
	sys.path.append('../auxiliary_functions/')
if '../../auxiliary_functions/' not in sys.path:
	sys.path.append('../../auxiliary_functions/')
from contourFunctions import *
from thresholdingFunctions import *


"""
multiThresholding Function:
Arguments:
	8bit Image File
	# of k-means centers
	pixelThresh: If True, thresholds based on the average greyscale value of the two darkest k-means. If False, thresholds by condidering the pixels assigned to the darkest k-means cluster.
Returns binary image, contoured image
"""
def multiThresholding(image, kthresh, pixelThresh = False):
	Z = image.reshape((-1,1))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #directly copied from opencv documentation
	ret,label,center=cv2.kmeans(Z,kthresh,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS) #sum of squared errors, labels, greyscale centers

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	sortedCenters = sorted(center.flatten())
	res = center[label.flatten()]
	kthreshed = res.reshape((image.shape))

	#now threshold the k-means clustered image to only keep the darkest cluster
	if pixelThresh:
		thresh_val = (sortedCenters[1] + sortedCenters[0])/2
		#thresh_val = sortedCenters[0]
		ret,proc = cv2.threshold(image,thresh_val,255,cv2.THRESH_BINARY)

	else:
		thresh_val = sortedCenters[0]
		ret, proc = cv2.threshold(kthreshed,thresh_val,255,cv2.THRESH_BINARY) #greyscale threshold, binary image

	frame_contours = []	

	contourImage = drawShapes(proc, image)
	return proc, contourImage

"""
SegmentingThresholding Function:
Arguments:
	8bit Image File
	# of Rows to be windowframed
	# of Columns to be windowframed
	# of k-means centers
	filename
	Filters to be sequentially applied, with the names of the filtering functions stored in a list (in order of filters to be applied). If None, no filter is used. 
	pixelThresh: If True, thresholds based on the average greyscale value of the two darkest k-means. If False, thresholds by condidering the pixels assigned to the darkest k-means cluster.
	Uniform segmenting boolean
	Dimensional factor (for non-uniform segmentation)
Returns binary image, image with contours
"""
def segmentedThresholding(img, rows, columns, kthresh, file, filter = None, save = False, pixelThresh = False, uni = True, factor = 1):

	
	# Selects the proper segmentation method and segments image into frames
	if uni:
		frames = windowFrame(img, rows, columns,  save, file, fac=0)
	else:
		frames = windowFrame(img, rows, columns,  save,  file, fac=factor)

	# Iteratively performs thresholding on each frame
	frames_contours = []
	for i in range(len(frames)):
		filteredFrame = frames[i]
		if filter != None:
			for fil in filter:
				filteredFrame = fil(filteredFrame)
		frame_shapes, contourImage = multiThresholding(filteredFrame, kthresh, pixelThresh = pixelThresh)
		frames_contours.append(frame_shapes)
	
	# Saves contoured frames (if selected)
	if save:
		os.system('mkdir frameContours_'+file[0:-4])
		for i in range(len(frames_contours)):
			cv2.imwrite('frameContours_' +file[0:-4] + "/frame"+str(i)+".tif",frames_contours[i])
	
	# Reconstructs frame contours into full image contours
	reconstructedImage = reconstructImage(frames_contours, uni, columns, rows)

	# Draws contours onto image
	allContours = drawShapes(reconstructedImage, img)
	
	# Returs contoured image and array of contours
	return reconstructedImage, allContours






