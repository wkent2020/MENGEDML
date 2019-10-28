#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:54:44 2019

@author: williamkent
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
if '/Users/williamkent/MENGEDML/individual_developers/will' not in sys.path:
	sys.path.append('/Users/williamkent/MENGEDML/individual_developers/will')
if '/Users/williamkent/MENGEDML/kmeans_thresholding/' not in sys.path:
	sys.path.append('/Users/williamkent/MENGEDML/kmeans_thresholding/')
from segmentedThresholding import *



direct = '../../test_images/'
file = "121.tif"
img = cv2.imread(direct + file,-1)
img = cropImage(img)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)


cols = 3
fac = 0.4
k = 3
savefile=False
binaryImage, contourImage = segmentedThresholding(img, 1, cols, k, file, uni=False, save=savefile, factor=fac)
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