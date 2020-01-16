#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:54:44 2019

@author: williamkent
"""

import cv2
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../..")
from kmeans_thresholding.segmentedThresholding import *
from shape_information.Shape import *
from image_processing.cropping.cropImage import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def get_hist_edge(freq, x, elist, lim, left=True):
	if left:
		i = int(len(freq)*0.4)
		while freq[i] > lim:
			i -= 1
			if i < 0:
				break
		if i >= 0:
			elist.append(x[i])
	else:
		i = int(len(freq)*0.66)
		while freq[i] > lim:
			i += 1
			if i == len(freq)-1:
				break
		if i != len(freq)-1:	
			elist.append(x[i+1])

def linetoangle(left,right):
	'''
	linetoangle function: takes linear regression information 
		and returns tuple of spray angle, left side of spray angle,
		and right side of spray angle. 
	'''
	left = np.arctan(1/left[0])
	right = np.arctan(1/right[0])
	opening = left+right
	return (opening,left,right)
	
def contours_to_points(contours, img, regpoints, method='simple', data='centers',plot=False, save=False):
	'''
	
	'''
	top = len(img)
	width = len(img[0])
	newlocs = []
	if data == 'centers':
		clocs = []
	#	contours.reverse()
		# Gets locations of droplets 
		for i in range(len(contours)):
			# Initializes contour as shape image to obtain info
			temp = Shape(contours[i], img)
			# If takign center of droplets
			if data == 'centers':
				clocs.append(temp.centerLocation)
				
				
		# Separates x and y values
		conty = [c[0] for c in clocs]
		contx = [c[1] for c in clocs]
		
		# Corrects vertical orientation
		
	#	conty = [top-c for c in conty]
		
		# Combines corrected y-values with x-values
		for i in range(len(conty)):
			newlocs.append([contx[i],conty[i]])
			
	elif data == 'black':
		for i in range(top):
			for j in range(width):
				if img[i][j] == 0:
					newlocs.append([j,i])
		
		conty = [c[1] for c in newlocs]
		contx = [c[0] for c in newlocs]
					
#		
	intervals = [0]
	intermid = []
	div_val = int(top/regpoints)
	
	# Computes vertical divisions
	for i in range(regpoints):
		intervals.append(div_val*(i+1))
		intermid.append(div_val*(i+1)-(div_val*(i+1)-intervals[i])/2)

	
	# Creates nested list for points
	separated_contours = []
	for i in range(regpoints):
		separated_contours.append([])
	
	# Separates points into vertical intervals
	for i in range(regpoints):
		for j in range(len(newlocs)):
			if (newlocs[j][1] >= intervals[i]) and (newlocs[j][1] <= intervals[i+1]):
				separated_contours[i].append(newlocs[j][0])
		
	left = []
	right = []
	if method == 'simple':
		for i in range(regpoints):
			mean = np.mean(separated_contours[i])
			std = np.std(separated_contours[i])
			left.append(mean - 1.5*std)
			right.append(mean + 1.5*std)
	
	if method == 'hist':
		lcount = 0
		rcount = 0
		if data == 'black':
			limstart = 20
		else:
			limstart = 3
		for i in range(regpoints):
			freq, x = np.histogram(separated_contours[i], bins=50)
			lim = limstart
			
			while len(left) == lcount:
				get_hist_edge(freq, x, left, lim)
				lim += 1
			lim = limstart
			lcount += 1
			while len(right) == rcount:
				get_hist_edge(freq, x, right,lim,left=False)
				lim += 1
			rcount += 1
	
	
	
	leftedge = linregress(left,intermid)
	rightedge = linregress(right,intermid)
	
	sangle, langle,rangle = linetoangle(leftedge,rightedge)
	
	print("The spray angle is %f" % sangle)
	print("The left angle is %f" % langle)
	print("The right angle is %f" % rangle)
	
	
	
	linex = np.linspace(0,width,100)
	leftline = [(leftedge[0]*x)+leftedge[1] for x in linex]
	rightline = [(rightedge[0]*x)+rightedge[1] for x in linex]
	
#	interp1d = 
	
	
	if plot:
		plt.close('all')	
		
		fig = plt.figure(figsize=(10,7))
		
		plt.subplot(2,2,1)
		plt.scatter(contx,conty, color='blue',s=0.1)
		plt.scatter(right,intermid, color='red')
		plt.scatter(left,intermid, color='black')
		plt.plot(linex,leftline,'k--')
		plt.plot(linex,rightline,'r--')
		pyplot.axis([0, width, 0, top])
		pyplot.gca().set_aspect('equal', adjustable='box')
		plt.title("Scatter-plot of Spray")
		
		plt.subplot(2,2,2)
		plt.imshow(img,cmap='gray')
		plt.scatter(right,intermid, color='red')
		plt.scatter(left,intermid, color='black')
		plt.plot(linex,leftline,'k--')
		plt.plot(linex,rightline,'r--')
		plt.ylim(0,top)
		plt.xlim(0,width)
		plt.title("Contoured Imaged")
		
		plt.subplot(2,2,3)
		plt.hist(separated_contours[1],50)
		plt.title("Histogram of x locations, Row 2")

		


direct = '../../test_images/'
file = "128.tif"
img = cv2.imread(direct + file,-1)
img = cropImage(img,cropTop=100, cropBottom = 0, cropLeft = 0, cropRight =0)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = np.uint8)


cols = 5
fac = 0.1
k = 3
unif = False
savefile=False
binaryImage, contourImage, contours = segmentedThresholding(img_rescaled, 1, cols, k, file, uni=unif, save=savefile, factor=fac)



#plt.close('all')
#plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='k')
#plt.subplot(1,1,1)
#plt.imshow(contourImage)


contours_to_points(contours, binaryImage, 6, method='hist', data='black', plot=True, save=False)


