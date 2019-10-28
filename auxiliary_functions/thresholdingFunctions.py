import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cropImage(image, cropTop=0, cropBottom = 0, cropLeft = 0, cropRight =0):
	"Crop pixels off the image"
	cropped_image = np.copy(image)
	cropped_image = cropped_image[cropTop:,cropLeft:]
	if cropBottom:
		cropped_image = cropped_image[:-cropBottom,]
	if cropRight:
		cropped_image = cropped_image[:,-cropRight]
	return cropped_image


def windowFrame(image, rows, columns,  save = True, file='none',fac=0):
	'''
	Separataes an image into segments. Called by segmentedThresholding in
	MENGEDML/kmeans_thresholding/segmentedThresholding.py 
	
	
	Input: cv2 image object, number of rows, number of columns, dilliniation factor
	       save boolean, file number. 
	
	Outputs: array of cv2 frame elements (frames of the image to be thresholded)
	
	'''
	
	frames = []
	lenRows, lenColumns = image.shape
	
	if fac != 0:
		cit = int((columns/2))
		rit = int((rows/2))
		
		columnvals = compute_dimensions(columns, cit,fac,lenColumns)
		rowvals = compute_dimensions(rows, rit,fac,lenRows)
		
		
		for i in range(len(rowvals)-1):
			for j in range(len(columnvals)-1):
				picture = image[int(rowvals[i]): int(rowvals[i+1]), int(columnvals[j]) : int(columnvals[j+1])]
				frames.append(picture)
	else:
		for i in range(rows):
			for j in range(columns):
				picture = image[int(lenRows / rows) * i: int(lenRows / rows) * (i+1), int(lenColumns / columns) * j: int(lenColumns / columns) * (j+1)]
				frames.append(picture)
    
	if save:
		os.system('mkdir windowFrames_'+file[0:-4])
		for i in range(len(frames)):
			cv2.imwrite('windowFrames_' +file[0:-4] + "/frame"+str(i)+".tif",frames[i])
#	print("The number of frames is %i" % len(frames))
	return frames


def reconstructImage(frames_contours, unif, columns, rows):
	'''
	Reconstructs an image from segments. Called by segmentedThresholding in
	MENGEDML/kmeans_thresholding/segmentedThresholding.py 
	
	Input: frames of contours, unform width boolean, number of columns, 
	       number of rows. 
	
	Outputs: reconstructed contour image
	
	'''
	
	i = 0
	if unif:
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
	else:
		while i < (len(frames_contours)):
			for j in range(columns):
				if j == 0:
					buildingRow = frames_contours[i]
					continue
				if j <= columns:
					buildingRow = np.hstack((buildingRow, frames_contours[i+j]))
	
			if i ==0:
				buildingColumn = buildingRow.copy()
			else:
				buildingColumn = np.vstack((buildingColumn, buildingRow))
				
			i += (columns)
			
	return buildingColumn


def adaptiveThresholding(image, thresholdType = 1, blockSize = 11, subtract = 2):
	'''
	Applies adaptive thresholding to image with either mean or Gaussian thresholding
	thresholdType of true gives adaptive mean thresholding and false gives adaptive Gaussian thresholding
	blockSize sets the size of the neighborhood, and subtract reduces the threshold by the given amount
	'''
	if thresholdType:
		thresh = cv2.ADAPTIVE_THRESH_MEAN_C
	else:
		thresh = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

	proc = cv2.adaptiveThreshold(image, 255, thresh, \
		cv2.THRESH_BINARY, blockSize, subtract)

	return proc


def compute_dimensions(columns, cit,factor,lenColumns):
	'''
	Computes the locations of boundaries between columns and rows. 
	Function is worded in terms of columns, but it works for rows as well.
	
	Called by windowFrame in this file. 
	
	Inputs: number of rows/columns, iterations, factor, width or length of image
	
	Output: Array of dilliniation values
	'''
	colfac = 0
	columnvals = [0]
	
	if columns%2 == 0:
		for i in range(cit):
			colfac += factor**i
		
		# Computs the width of first column
		column1 = lenColumns/(2*colfac)
		colend = 0
		
		# Iteratively computes addresses of the right of the column
		for i in range(cit):
			colend += column1*(factor**i)
			columnvals.append(colend)
			
		for i in range(cit):
			colend += (columnvals[cit-i]-columnvals[cit-i-1])
			columnvals.append(colend)
	elif columns == 1:
		columnvals.append(lenColumns)
	else:
		for i in range(cit+1):
			if i == cit:
				colfac += 0.5*(factor**i)
			else:
				colfac += factor**i
		# Computs the width of first column
		column1 = lenColumns/(2*colfac)
		colend = 0
		
		# Iteratively computes addresses of the right of the column
		for i in range(cit+1):
			colend += column1*(factor**i)
			columnvals.append(colend)
			
		for i in range(cit):
			colend += (columnvals[cit-i]-columnvals[cit-i-1])
			columnvals.append(colend)
			
	return columnvals

def drawShapes(image_binarized, image):
	'''
	Draw contours onto images
	'''
	contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	shapes_image = np.copy(image)
	
#	print("There are %d contours" % len(contours))
	#change back to RGB for easier visualization
	shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
	shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
	return shapes_image