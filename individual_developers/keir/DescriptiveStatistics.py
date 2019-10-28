import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import os
from segmentedThresholding import *

def histogram(image):
	Z = np.concatenate(np.float32(image.reshape((-1,1))))
	img_max = image.max()
	img_min = image.min()
	bins = img_max - img_min
	fig, axs = plt.subplots(1, 1)
	axs.hist(Z, bins)
	#axs.set_xlim(0,255)
	plt.show()

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
		r1, r2, c1, c2 = shape.boundary
		plt.imshow(image[r1:r2, c1:c2]) #false positive: image[8:21, 241:288]
		plt.show()

plt.scatter(shapesDensity, shapesArea)
plt.show()
plt.hist(shapesArea, density=False, bins=30)
plt.show()




