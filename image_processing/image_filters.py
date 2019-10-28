#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:54:33 2019

@author: williamkent
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def histogram(image, show = True, save = ''):
	Z = np.concatenate(np.float32(image.reshape((-1,1))))
	img_max = image.max()
	img_min = image.min()
	bins = img_max - img_min
	fig, axs = plt.subplots(1, 1)
	axs.hist(Z, bins)
	#axs.set_xlim(0,255)
	if show:
		plt.show()
	if len(save) != 0:
		os.system('mkdir savedHistograms')
		plt.savefig('savedHistograms/' + save)
		plt.close()

def canny(img, plot = False):
	edges = cv2.Canny(img,100,40)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
		plt.show()
	return edges

def bilateralFilter(img, plot = False):
	blur = cv2.bilateralFilter(img,9,75,75)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(blur,cmap = 'gray')
		plt.title('Bilateral Filtered'), plt.xticks([]), plt.yticks([])
		plt.show()
	return blur

def medianFilter(img, plot = False):
	median = cv2.medianBlur(img,5)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(median,cmap = 'gray')
		plt.title('Median Filtered'), plt.xticks([]), plt.yticks([])
		plt.show()
	return median

def gaussianFilter(img, plot = False):
	blur = cv2.GaussianBlur(img,(5,5),0)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(blur,cmap = 'gray')
		plt.title('Gaussian Filtered'), plt.xticks([]), plt.yticks([])
		plt.show()
	return blur

def averageBlur(img, plot = False):
	blur = cv2.blur(img, (5,5))
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(blur,cmap = 'gray')
		plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
		plt.show()
	return blur

#blur = gaussianFilter(img)
#histogram(blur)

def histogramEqualization(img):
	equ = cv2.equalizeHist(img)
	return equ

def adaptiveHistogram(img, plot = False):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(cl1,cmap = 'gray')
		plt.title('Adaptive Histogram'), plt.xticks([]), plt.yticks([])
		plt.show()
	return cl1

def laplace(img, plot = False):
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	if plot:
		plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
		plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
		plt.show()
	return laplacian

def sobelx(img, plot = False):
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	if plot:
		plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
		plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	return sobelx

def sobely(img, plot = False):
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	if plot:
		plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(2,2,2),plt.imshow(sobely,cmap = 'gray')
		plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	return sobely