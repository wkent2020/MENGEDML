import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import sys

sys.path.append('Normalization_Thresholding/')
from .Shape import Shape, NestedShapes, imagePad
from .normalizeImages import Normalize, cropImage
from .iterativeGrowth import iterativeGrowthThresholder

sys.path.append('Macroscopic_Analysis/')
from .sprayDensities import numberDensity_byrows, massDensity_byrows, getSectorRadiusDensities, areaDensity_byrows
from .sprayEdges import contours_to_points
from .nozzlePosition import getNozzlePosition

def getImages():
	imgs = np.arange(1, 399, 1)
	images = []
	for i in imgs:
		try:
			images.append(cropImage(cv2.imread('../80bar/T2-X=0.00_Y=1.00__000'+str(i)+".tif", -1), 100))
		except:
			try:
				images.append(cropImage(cv2.imread('../80bar/T2-X=0.00_Y=1.00__00'+str(i)+".tif", -1), 100))
			except:
				try:
					images.append(cropImage(cv2.imread('../80bar/T2-X=0.00_Y=1.00__0'+str(i)+".tif", -1), 100))
				except:
					try:
						images.append(cropImage(cv2.imread('../80bar/T2-X=0.00_Y=1.00__'+str(i)+".tif", -1), 100))
					except:
						print('No Image Found')
	return images

def getSprayImages(images):
	images_mean = [np.mean(i) for i in images] 
	images_STDs = [np.std(i) for i in images]
	min_STD = np.min(images_STDs)
	max_STD = np.max(images_STDs)
	sprayImagesIndices = []
	for i in range(len(images)):
		if images_STDs[i] > (min_STD + max_STD) / 2.0 :
			sprayImagesIndices.append(i)
	sprayImages = [images[i] for i in sprayImagesIndices]
	return sprayImages

def shiftImageMean(image, ensembleMean):
	mean = np.mean(image)
	shift = ensembleMean - mean
	return image + shift

def analyzeImageByIndex(normalizedSprayImages, index, thresholdMode = 'Ensemble'):

	"""
	------------------------------------
	Image Thresholding
	------------------------------------
	"""
	image = normalizedSprayImages[index]

	if thresholdMode == 'Ensemble':
		ensembleMean = np.mean(normalizedSprayImages)
		ensembleSTD = np.std(normalizedSprayImages)
		ensembleSigma_list = np.array([2.5, 1.5, 1.0, 0.75, 0.6, 0.5, 0.45, 0.42, 0.41, 0.40]) * -1.0
		thresholdList = list((ensembleSigma_list * ensembleSTD) + ensembleMean)
		image = shiftImageMean(image, ensembleMean)
		image_mean = np.mean(image)
		image = imagePad(image, 3,3,3,3, greyscale = image_mean)

	elif thresholdMode == 'Individual':
		image_mean = np.mean(image)
		image_SD = np.std(image)
		sigma_list = np.array([2.5, 1.5, 1.0, 0.75, 0.6, 0.5, 0.45, 0.42, 0.41, 0.40]) * -1.0 #selected standard deviations from mean
		thresholdList = list(sigma_list * image_SD + image_mean)
		image = imagePad(image, 3,3,3,3, greyscale = image_mean)
	
	
	#dropletDataFrame, holeContoursDF, binaryImage = iterativeGrowthThresholder(image.copy(), absoluteThresholdList = thresholdList, plot = False)
	dropletDataFrame, holeContoursDF, binaryImage, viewImage = iterativeGrowthThresholder(image.copy(), absoluteThresholdList = thresholdList, plot = True)
	

	"""
	---------------------------------------------------
	Adding macroscopic variables to contour information
	Includes:
		Distance from Nozzle (Radius)
		Angle from Nozzle (Angle)
	---------------------------------------------------
	"""

	#get individual nozzle position
	leftRegression, rightRegression, linex, leftline, rightline, intersectionX, intersectionY = contours_to_points(binaryImage.copy(), dropletDataFrame['Row COM'], dropletDataFrame['Column COM'], 6, method='hist', data='black', plot=False, save=False)
	
	#get ensemble nozzle position using database of all binary images and dataframes
	#intersectionX, intersectionY = getNozzlePosition(binaryImagesList, rowLOC_list, columnLOC_list)

	dropletDataFrame['Distance From Nozzle'] = np.sqrt((np.array(dropletDataFrame['Row COM']) - intersectionY)**2.0 + (np.array(dropletDataFrame['Column COM']) - intersectionX)**2.0)

	angle_contour = np.array(np.arccos((np.array(dropletDataFrame['Row COM']) - intersectionY) / np.sqrt((np.array(dropletDataFrame['Column COM'])-intersectionX)**2 + (np.array(dropletDataFrame['Row COM']) - intersectionY)**2))) * (180.0 / np.pi)
	angle_sign = (np.array(dropletDataFrame['Column COM']) - intersectionX) / np.abs((np.array(dropletDataFrame['Column COM']) - intersectionX))
	dropletDataFrame['Angle'] = angle_contour*angle_sign

	"""
	------------------
	DENSITY ANALSYIS
	------------------
	"""
	sectors_massDensity, sectors_areaDensity, sectors_numberDensity, \
		radius_massDensity, radius_areaDensity, radius_numDensity, \
		radii, angles = getSectorRadiusDensities(image.copy(), binaryImage.copy(), dropletDataFrame['Row COM'], dropletDataFrame['Column COM'], intersectionX, intersectionY, radiiNumber = 40, thetaIncrement = 5, maximumPixel = 255)
	
	sectors_DispersityDensity = sectors_numberDensity / sectors_areaDensity
	radial_DispersityDensity = radius_numDensity / radius_areaDensity

	numberDensity = numberDensity_byrows(image.copy(), dropletDataFrame['Row COM'], dropletDataFrame['Column COM'], 20, 20, plot = False)
	massDensity, massImage = massDensity_byrows(image.copy(), binaryImage.copy(), 20, 20, maximumPixel = 255, plot = False)
	areaDensity = areaDensity_byrows(binaryImage.copy(), 20, 20, plot = False)
	dispersityDensity = numberDensity/areaDensity

	return dropletDataFrame, binaryImage, viewImage, \
		sectors_massDensity, sectors_areaDensity, sectors_numberDensity, sectors_DispersityDensity, \
		radius_massDensity, radius_areaDensity, radius_numDensity, radial_DispersityDensity, \
		massDensity, areaDensity, numberDensity, dispersityDensity, \
		radii, angles  


def main():
	#Get Images
	images = getImages()

	#Normalize Images
	images = Normalize(images, 2**8, [0,10], [100,200,412,512], mode = 'DBB')
	
	#Choose only images containing spray
	sprayImages = getSprayImages(images)

	#analyze a given image by the index in sprayImages--here, 0 is chosen (the first image in sprayImages)
	dropletDataFrame, binaryImage, viewImage, \
		sectors_massDensity, sectors_areaDensity, sectors_numberDensity, sectors_DispersityDensity, \
		radius_massDensity, radius_areaDensity, radius_numDensity, radial_DispersityDensity, \
		massDensity, areaDensity, numberDensity, dispersityDensity, \
		radii, angles  = analyzeImageByIndex(sprayImages, 0, thresholdMode = 'Ensemble')
	
	plt.imshow(viewImage, cmap = 'gray')
	plt.show()

	def plotSectorsDensity(densityMap, radii, angles):
		rows = np.linspace(0,densityMap.shape[0], densityMap.shape[0])
		cols = np.linspace(0,densityMap.shape[1], densityMap.shape[1])
		contourPlot = plt.contourf(angles, radii, densityMap, cmap = 'inferno')
		plt.colorbar(contourPlot, aspect=10)
		plt.ylim(np.max(radii), np.min(radii))
		plt.show()

	plotSectorsDensity(sectors_massDensity, radii, angles)
	plotSectorsDensity(sectors_areaDensity, radii, angles)
	plotSectorsDensity(sectors_numberDensity, radii, angles)
	plotSectorsDensity(sectors_DispersityDensity, radii, angles)

	plt.imshow(sectors_DispersityDensity, cmap='viridis')
	plt.show()

	plt.imshow(massDensity)
	plt.show()

	plt.imshow(areaDensity)
	plt.show()

	plt.imshow(numberDensity)
	plt.show()

	plt.imshow(dispersityDensity)
	plt.show()

	"""
	------------------------------------
	Example Statistical Analysis of Droplet Features
	------------------------------------

	plt.hist(np.array(dropletDataFrame['Angle']), bins = 'auto')
	plt.show()

	plt.scatter(np.array(dropletDataFrame['Angle']), np.array(dropletDataFrame['Radius of Gyration (Unweighted)']))
	plt.show()

	plt.scatter(np.array(dropletDataFrame['Angle']), np.array(dropletDataFrame['Shape Area']))
	plt.show()

	plt.scatter(np.array(dropletDataFrame['Angle']), np.array(dropletDataFrame['Outer Perimeter']))
	plt.show()
	"""

# main()


