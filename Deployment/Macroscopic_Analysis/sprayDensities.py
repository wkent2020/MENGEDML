import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sprayEdges import *

def numberDensity_byrows(image, rowLOC, columnLOC, rows, columns, plot = False):
	counts, col, row = np.histogram2d(columnLOC, rowLOC, [columns, rows])
	counts = counts.T

	if plot:
		plt.imshow(counts, cmap='viridis')
		plt.show()

		plt.hist2d(columnLOC, rowLOC, bins = [columns, rows])
		plt.ylim(len(image[:,0]), 0)
		plt.show()
		rows = np.linspace(0,len(image[0,:]),rows)
		cols = np.linspace(0,len(image[:,0]),columns)
		plt.contourf(cols, rows, counts)
		plt.ylim(len(image[:,0]), 0)
		plt.show()
		rowNumDen = []
		for i in range(counts.shape[0]):
			rowNumDen.append(np.sum(counts[i,:]))
		plt.bar(np.arange(len(rowNumDen)) + 1, rowNumDen)
		plt.show()
	return counts

def massDensity_byrows(image, binaryImage, rows, columns, maximumPixel = 255, plot = False):
	massImage = image.copy()
	massImage[binaryImage == 255] = maximumPixel
	massImage = maximumPixel - massImage #now, darker pixels have higher values

	massArray = np.zeros(rows*columns).reshape(rows, columns)
	lenRows, lenColumns = massImage.shape
	for i in range(rows):
		for j in range(columns):
			frame = massImage[int(lenRows / rows) * i: int(lenRows / rows) * (i+1), int(lenColumns / columns) * j: int(lenColumns / columns) * (j+1)]
			massArray[i][j] = np.sum(frame)

	if plot:
		plt.imshow(massArray, cmap='viridis')
		plt.show()
		rows = np.linspace(0,len(massImage[:,0]),rows)
		cols = np.linspace(0,len(massImage[0,:]),columns)
		plt.contourf(cols, rows, massArray)
		plt.ylim(len(massImage[:,0]), 0)
		plt.show()
		rowMass = []
		for i in range(massArray.shape[0]):
			rowMass.append(np.sum(massArray[i,:]))
		plt.bar(np.arange(len(rowMass)) + 1, rowMass)
		plt.show()

	return massArray, massImage

def areaDensity_byrows(binaryImage, rows, columns, plot = False):
	areaImage = binaryImage.copy()
	areaImage[binaryImage == 0] = 1
	areaImage[binaryImage == 255] = 0

	areaArray = np.zeros(rows*columns).reshape(rows, columns)
	lenRows, lenColumns = areaImage.shape
	for i in range(rows):
		for j in range(columns):
			frame = areaImage[int(lenRows / rows) * i: int(lenRows / rows) * (i+1), int(lenColumns / columns) * j: int(lenColumns / columns) * (j+1)]
			areaArray[i][j] = np.sum(frame)

	if plot:
		plt.imshow(areaArray, cmap='viridis')
		plt.show()
		rows = np.linspace(0,len(areaImage[:,0]),rows)
		cols = np.linspace(0,len(areaImage[0,:]),columns)
		plt.contourf(cols, rows, areaArray)
		plt.ylim(len(areaImage[:,0]), 0)
		plt.show()
		rowMass = []
		for i in range(areaArray.shape[0]):
			rowMass.append(np.sum(areaArray[i,:]))
		plt.bar(np.arange(len(rowMass)) + 1, rowMass)
		plt.show()

	return areaArray


def densityDifference(density1, density2):
	return density2 - density1


def getRadii(binaryImageSample, nozzleX, nozzleY, radiiNumber):
	#determine the maximum distance of a pixel from the average nozzle, then define radii thickness using that distance and radiiNumber
	imageGrid = np.indices((binaryImageSample.shape[0], binaryImageSample.shape[1]))
	row_grid, col_grid = imageGrid[0], imageGrid[1]

	deltaRow_squared = (row_grid - nozzleY)**2.0
	deltaCol_squared = (col_grid - nozzleX)**2.0

	distanceGrid = np.sqrt(deltaRow_squared + deltaCol_squared)
	maxDistance = np.where(distanceGrid == np.max(distanceGrid))
	maxRow, maxCol = maxDistance[0][0], maxDistance[1][0]
	minDistance = np.where(distanceGrid == np.min(distanceGrid))
	minRow, minCol = minDistance[0][0], minDistance[1][0]

	radii = np.linspace(np.min(distanceGrid), np.max(distanceGrid), radiiNumber)

	return radii


def getSectorRadiusDensities(image, binaryImage, rowLOC, columnLOC, intersectionX, intersectionY, radiiNumber = 20, thetaIncrement = 5, maximumPixel = 255):

	#get sectors
	radii = getRadii(binaryImage, intersectionX, intersectionY, radiiNumber)
	angles = np.arange(-thetaIncrement*9, thetaIncrement*10, thetaIncrement)

	densityContainer = np.zeros(len(radii)*len(angles)).reshape(len(radii), len(angles))

	#Now, caluculate various density profiles
	sectors_massDensity = densityContainer.copy()
	sectors_areaDensity = densityContainer.copy()
	sectors_numberDensity = densityContainer.copy()

	massImage = image.copy()
	massImage[binaryImage == 255] = maximumPixel
	massImage = maximumPixel - massImage #now, darker pixels have higher values, and non-contoured pixels are 0

	areaImage = binaryImage.copy()
	areaImage[areaImage == 0] = 1
	areaImage[areaImage == 255] = 0 #now, thresholed pixels are 1 and background is 0

	#Summing Mass and Area Densities
	def distance(x1, y1, x2, y2):
		return np.sqrt((y2 - y1)**2 + (x2-x1)**2)

	radius_massDensity = np.zeros(len(radii)) #only for radius
	radius_areaDensity = np.zeros(len(radii)) #only for radius
	for i in range(massImage.shape[0]):
		for j in range(massImage.shape[1]):
			radius = distance(intersectionX, intersectionY, j, i)
			angle = np.arccos((i - intersectionY) / np.sqrt((j-intersectionX)**2 + (i-intersectionY)**2)) * (180.0 / np.pi)
			if j < intersectionX:
				angle *= -1.0
			radius_classing = np.argmin(np.abs(radii - radius))
			angle_classing = np.argmin(np.abs(angle - angles))

			radius_massDensity[radius_classing] += massImage[i,j]
			radius_areaDensity[radius_classing] += areaImage[i,j]

			sectors_massDensity[radius_classing, angle_classing] += massImage[i,j]
			sectors_areaDensity[radius_classing, angle_classing] += areaImage[i,j]

	#Summing Number Densities
	radius_numDensities = np.zeros(len(radii))
	radius_contour = np.array(distance(intersectionX, intersectionY, columnLOC, rowLOC))
	angle_contour = np.array(np.arccos((np.array(rowLOC) - intersectionY) / np.sqrt((np.array(columnLOC)-intersectionX)**2 + (np.array(rowLOC) - intersectionY)**2))) * (180.0 / np.pi)
	angle_sign = (np.array(columnLOC) - intersectionX) / np.abs((np.array(columnLOC) - intersectionX))
	angle_contour = angle_contour*angle_sign

	angle_classings = np.zeros(len(radius_contour))
	radii_classings = np.zeros(len(radius_contour))

	for i in range(len(radius_contour)):
		angle_classing = np.argmin(np.abs(angles- angle_contour[i]))
		angle_classings[i] = angle_classing
		classing = np.argmin(np.abs(radii - radius_contour[i]))
		radii_classings[i] = classing
		radius_numDensities[classing] += 1
		sectors_numberDensity[classing, angle_classing] += 1

	return sectors_massDensity, sectors_areaDensity, sectors_numberDensity, radius_massDensity, radius_areaDensity, radius_numDensities, radii, angles

