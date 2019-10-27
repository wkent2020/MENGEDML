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

def cropContour(contour, image, border = 0):
	'''
	Crop shape from rest of image
	'''

	boundary = contourBoundary(contour) 
	# create a single channel pixel white image
	canvas = np.zeros(image.shape).astype(image.dtype) + 255
	fill = cv2.fillPoly(canvas, pts =[contour], color=0)
	#keep shape in grayscale, turn background white
	anti_fill = cv2.bitwise_or(image,fill)
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
    




