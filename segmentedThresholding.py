import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from thresholdingFunctions import *

direct = 'Old_Groups_Code/norm_imgs/'
file = "120.tif"
img = cv2.imread(direct + file,-1)
cropTop = 0
img = cv2.imread(direct + file,-1)[cropTop:]
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)



def multiThresholding(image, kthresh, pixelThresh = False):
	Z = image.reshape((-1,1))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #directly copied from opencv documentation
	ret,label,center=cv2.kmeans(Z,kthresh,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS) #sum of squared errors, labels, greyscale centers
#	print(center.flatten())
	#print(ret)
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


def segmentedThresholding(img, rows, columns, kthresh, filter = None, save = False, pixelThresh = False, frame2 = False, rmdiv=0, cmdiv=0):

	if frame2:
		frames = windowFrame2(img, rows, columns, rmdiv, cmdiv, save, file)
	else:
		frames = windowFrame(img, rows, columns, save, file)
	
	frames_contours = []
	for i in range(len(frames)):
		filteredFrame = frames[i]
		if filter != None:
			for fil in filter:
				filteredFrame = fil(filteredFrame)
		frame_shapes, contourImage = multiThresholding(filteredFrame, kthresh, pixelThresh = pixelThresh)
		frames_contours.append(frame_shapes)
		#histogram(framesi[], show = False, save = 'frame' + str(i))
	
	if save:
		os.system('mkdir frameContours_'+file[0:-4])
		for i in range(len(frames_contours)):
			cv2.imwrite('frameContours_' +file[0:-4] + "/frame"+str(i)+".tif",frames_contours[i])
	
	#recombines segmented binary files into single binary image
	if frame2:
		i = 0
		while i < (len(frames_contours)):
			for j in range(columns+cmdiv-1):
				print((i,j))
				if j == 0:
					buildingRow = frames_contours[i]
					continue
				if (j+1) <= (columns+cmdiv):
					buildingRow = np.hstack((buildingRow, frames_contours[i+j]))
	
			if i ==0:
				buildingColumn = buildingRow.copy()
			else:
				buildingColumn = np.vstack((buildingColumn, buildingRow))
	
			i += (columns+cmdiv-1)
	else:
		i = 0
		while i < (len(frames_contours)):
			for j in range(columns):
				if j == 0:
					buildingRow = frames_contours[i]
					continue
				if (j+1) <= columns:
					buildingRow = np.hstack((buildingRow, frames_contours[i+j]))
	
			if i ==0:
				buildingColumn = buildingRow.copy()
			else:
				buildingColumn = np.vstack((buildingColumn, buildingRow))
	
			i += columns

	allContours = drawShapes(buildingColumn, img)
	return buildingColumn, allContours

def segmentedThresholding3(img, rows, columns, kthresh, filter = None, save = False, pixelThresh = False, frame2 = False, factor = 1):

	frames = windowFrame3(img, rows, columns, factor, save, file)

	
	frames_contours = []
	for i in range(len(frames)):
		filteredFrame = frames[i]
		if filter != None:
			for fil in filter:
				filteredFrame = fil(filteredFrame)
		frame_shapes, contourImage = multiThresholding(filteredFrame, kthresh, pixelThresh = pixelThresh)
		frames_contours.append(frame_shapes)
		#histogram(framesi[], show = False, save = 'frame' + str(i))
	
	if save:
		os.system('mkdir frameContours_'+file[0:-4])
		for i in range(len(frames_contours)):
			cv2.imwrite('frameContours_' +file[0:-4] + "/frame"+str(i)+".tif",frames_contours[i])
	
	#recombines segmented binary files into single binary image

	i = 0
	while i < (len(frames_contours)):
		for j in range(columns):
			if j == 0:
				buildingRow = frames_contours[i]
				continue
			if j <= columns:
				buildingRow = np.hstack((buildingRow, frames_contours[i+j]))
#			print("Column %i" % j)
#			print("Row %i" % i)

		if i ==0:
			buildingColumn = buildingRow.copy()
		else:
			buildingColumn = np.vstack((buildingColumn, buildingRow))
#		print(i)
		i += (columns)

	allContours = drawShapes(buildingColumn, img)
	return buildingColumn, allContours

#arguments: loaded 8bit image file, # rows, # columns, # of k-means clusters, filters to be applied (names of functions, in python list. If no filters to be applied, ),
"""
SegmentingThresholding Function:
Arguments:
	8bit Image File
	# of Rows to be windowframed
	# of Columns to be windowframed
	# of k-means centers
	Filters to be sequentially applied, with the names of the filtering functions stored in a list (in order of filters to be applied). If None, no filter is used. 
	pixelThresh: If True, thresholds based on the average greyscale value of the two darkest k-means. If False, thresholds by condidering the pixels assigned to the darkest k-means cluster.

Returns binary image, image with contours
"""

cols = 3
fac = 0.3
k = 3
savefile=False
binaryImage, contourImage = segmentedThresholding3(img, 1, cols, k, filter = None, pixelThresh = False,save=savefile,frame2 = True, factor=fac)
#binaryImage2, contourImage2 = segmentedThresholding(img, 1, cols, k, filter = None, pixelThresh = False,save=False)
#binaryImage3, contourImage3 = multiThresholding(img, k)
#binaryImage2, contourImage2 = segmentedThresholding3(img, 1, cols, 2, filter = None, pixelThresh = False,save=False,frame2 = True, factor=fac)
#binaryImage22, contourImage22 = segmentedThresholding(img, 1, cols, 2, filter = None, pixelThresh = False,save=False)
#binaryImage32, contourImage32 = multiThresholding(img, 2)

plt.close('all')
plt.figure(num=None, figsize=(11, 7), dpi=100, facecolor='w', edgecolor='k')
#plt.subplot(2,3,1)
#plt.imshow(contourImage32)
#plt.title(r"$k=2$ Uniform")
#plt.subplot(2,3,3)
#plt.imshow(contourImage2)
#plt.title("$k=2$ Segmented: %i Varied Columns" % int(cols))
#plt.subplot(2,3,2)
#plt.imshow(contourImage22)
#plt.title("$k=2$ Segmented: 3 Even Columns" )
#plt.subplot(2,3,4)
#plt.imshow(contourImage3)
#plt.title(r"$k=3$ Uniform")
#plt.subplot(2,3,6)
plt.imshow(contourImage)
plt.title("$k=3$ Segmented: %i Varied Columns" % int(cols))
#plt.subplot(2,3,5)
#plt.imshow(contourImage2)
#plt.title("$k=3$ Segmented: 3 Even Columns" )
#plt.tight_layout()
plt.show()
#plt.savefig("OPfigs/k3_multi.png", dpi=300)


