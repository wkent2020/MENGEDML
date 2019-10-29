import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def drawShapes(image_binarized, image):
    '''
    Draw contours onto images
    '''
    contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes_image = np.copy(image)

    for contour in contours:
        croppedContour = cropContour(contour, image)
        centroid = centerOfMass(croppedContour)
        print(centroid)

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
multiThresholding(img, 3)
