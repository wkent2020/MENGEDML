import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segmentedThresholding import getContours, Shape

def contourBoundary(contour):
	'''
	Save location of the shape in the context of the larger image
	'''

	x,y,w,h = cv2.boundingRect(contour)
	return np.array([y,y+h,x,x+w])

def fillPoly(contour, image):
	#Returns filled polygon with intensity 0 on white array of same size as original image

	canvas = np.zeros(image.shape).astype(image.dtype) + 255
	fill = cv2.fillPoly(canvas, pts =[contour], color=0)
	return fill

def cropContour(contour, image, border = 0):
	'''
	Crop shape from rest of image
	'''

	boundary = contourBoundary(contour) 
	#keep shape in grayscale, turn background white
	anti_fill = cv2.bitwise_or(image,fillPoly(contour, image))
	croppedContour = anti_fill[boundary[0]:boundary[1],\
					boundary[2]:boundary[3]]
	#also crop to slightly larger than boundary so shape isn't right at 
	#the edge of the image
	#this will be useful if we want to draw more contours on a shape after cropping it
	if border:
		borderedContour = anti_fill[boundary[0]-border:boundary[1]+border,\
					boundary[2]-border:boundary[3]+border]
		return borderedContour
	return croppedContour

def meanIntensity(contour, image):
    #Return mean intensity for a contour

    croppedContour = cropContour(contour, image)
    mask = np.logical_not(np.logical_not(croppedContour -255)).astype('uint8')
    return cv2.mean(croppedContour, mask= mask)[0]

def centerOfMass(croppedContour):
    '''
    Compute the center of mass relative to the contour coordinates
    '''
    #Convert the intensity to a float with 0 as light and 1 as dark
    inverseFloat = (croppedContour.astype(float) - 255)/-255 
    #Compute total mass
    mass = np.sum(inverseFloat) 
    grids = np.ogrid[[slice(0,i) for i in inverseFloat.shape]]
    centroid = [np.sum(inverseFloat * grids[dim].astype(float)) / mass 
                    for dim in range(inverseFloat.ndim)]
    return centroid

def bigDropDivision(croppedContour, buffer = 1):
	'''
	Breaks large droplets into smaller droplets
	'''

	#Equalize and find minimum
	contour = cv2.cvtColor(croppedContour,cv2.COLOR_GRAY2RGB)
	equalize = cv2.equalizeHist(croppedContour) 
	#Might be redundant given the equalization 
	#Probably the location of the minimum doesn't change after equalization
	intensityMin = np.amin(equalize) + buffer
	#Find peaks and background
	_,peaks = cv2.threshold(equalize,intensityMin,255,cv2.THRESH_BINARY_INV) 
	_,background = cv2.threshold(equalize,254,255,cv2.THRESH_BINARY_INV)
	unknown = cv2.subtract(background,peaks)

	'''
	#Print line debugging
	fig, axes = plt.subplots(1, 2)
	axes[0].set_title("Background")
	axes[0].imshow(background,cmap='gray')
	axes[1].set_title("Peaks")
	axes[1].imshow(peaks,cmap='gray')
	plt.show()
	plt.close()
	plt.imshow(unknown,cmap='gray')
	plt.show()
	'''

	#Find connected components
	_, markers = cv2.connectedComponents(peaks)
	markers += 1
	markers[unknown==255] = 0
	#Watershed
	markers = cv2.watershed(contour,markers)

	#Mark border in white
	contour[markers == -1] = [255, 255, 255]

	
	plt.close()
	fig, axes = plt.subplots(1, 2)
	axes[0].set_title("Contour")
	axes[0].imshow(contour,cmap='gray')
	axes[1].set_title("Markers")
	axes[1].imshow(markers,cmap='gray')
	plt.show()
	

	#Contour each droplet
	blobs = []
	canvas = np.zeros(croppedContour.shape).astype('uint16')+2
	for marker in range(markers.max()-2):
		binary = np.equal(markers, canvas).astype('uint8')*255
		'''
		plt.close()
		plt.imshow(binary,cmap='gray')
		plt.show()
		'''
		contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		blobs.append(np.array(contours)[0])
		canvas += 1
	blobs = np.array(blobs)

	#Convert back to gray
	contour = cv2.cvtColor(contour, cv2.COLOR_RGB2GRAY)

	return contour, markers, blobs

def descriptiveStatistics():

	img, binaryImage, image, contours = getContours()

	shapes = []
	shapesCrop = []
	shapesBoundary = []
	shapesArea = []
	shapesPeri = []
	shapesDensity = []
	centers = []

	for i in range(len(contours)):
		shape = Shape(contours[i])
		shapeCrop = shape.crop(img)
		shapeCropFlat = shapeCrop.flatten()
		shapeNoBackground  = list(filter(lambda x: x != 255, shapeCropFlat))
		shapeArea = len(shapeNoBackground)
		shapeDensity = np.sum(shapeNoBackground) / shapeArea

		if shapeArea > 2000 or shapeArea <= 5:
			continue

		shapes.append(shape)
		shapesCrop.append(shapeCrop)
		boundary = shape.boundary
		centers.append([(boundary[0] + boundary[1]) / 2, (boundary[2] + boundary[3]) / 2])

		shapesBoundary.append(boundary)
		shapesPeri.append(shape.peri)
		shapesDensity.append(shapeDensity)
		shapesArea.append(shapeArea)
		
		if shapeArea >500:
			plt.imshow(shapeCrop, cmap = 'gray')
			plt.show()
			print(shapeCrop)
			r1, r2, c1, c2 = shape.boundary
			plt.imshow(image[r1:r2, c1:c2]) #false positive: image[8:21, 241:288]
			plt.show()
			print(image[r1:r2, c1:c2])
			bigDropDivision(shapeCrop)

	plt.scatter(shapesDensity, shapesArea)
	plt.show()
	plt.hist(shapesArea, density=False, bins=30)
	plt.show()

def drawShapes(image_binarized, image):
	'''
	Draw contours onto images
	'''
	contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	shapes_image = np.copy(image)

	'''
	count = 0
	for contour in contours:
		croppedContour = cropContour(contour, image)
		if count in [32, 34, 56]: 
			print(count)
			print(croppedContour)
		numPeaks = bigDropDivision(croppedContour)
		if numPeaks > 7:
			print("numPeaks = " + str(numPeaks))
			print("count = " + str(count))
		count += 1
	'''

	#change back to RGB for easier visualization
	shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
	#	plt.imshow(shapes_image) # uncomment these lines to plot in real time
	#	plt.show()
	shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
	plt.imshow(shapes_image) 
	plt.show()

def multiThresholding(image, kthresh, kthcenter = 0, plotHistogram = False):

	Z = image.reshape((-1,1))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #directly copied from opencv documentation
	ret,label,center = cv2.kmeans(Z,kthresh,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	if plotHistogram:
		kMeansHistogram(Z, label, kthresh)
	print(center)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
    # Puts something into single array
	res = center[label.flatten()]
    # Takes 1-D array and puts back into image format
	kthreshed = res.reshape((image.shape))
    # Lowest grayscale k-means center: aka some color
	if(kthcenter == 0):
		thresh_val = min(center)[0]
	else:
		thresh_val = np.sort(np.concatenate(center))[kthcenter]
	print("kmeans multithreshold value of "+str(thresh_val))            
	#now threshold the k-means clustered image to only keep the darkest cluster
	ret,proc = cv2.threshold(kthreshed,thresh_val,255,cv2.THRESH_BINARY) #binarizes
    # Binary image file
	print(proc)
    # Threshold value
	print(ret)
	#plt.imshow(proc,cmap = 'gray')
	#plt.show()
	#plt.imshow(img, cmap = 'gray')
	#plt.show()
	#canny(proc)

	
	drawShapes(proc, img)

img = cv2.imread("individual_developers/jackson/120.tif",-1)
descriptiveStatistics()
multiThresholding(img, 3)
