import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from .Shape import Shape, NestedShapes
from .normalizeImages import Normalize

def getContours(image_binarized, image):
	'''
	Draw contours onto images
	'''
	image_binarized = np.array(image_binarized).astype('uint8') #adjustment

	contours, hierarchy = cv2.findContours(cv2.bitwise_not(image_binarized), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #default: cv2.CHAIN_APPROX_SIMPLE
	#shapes_image = np.copy(image) #adjustment
	shapes_image = np.array(image).astype('uint8') #adjustment

	#change back to RGB for easier visualization
	shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
	shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
	return shapes_image, contours, hierarchy

def multiThresholding(image, absoluteThreshold = None, excludePixels = None):
	Z = image.reshape((-1,1))
	Z = np.float32(Z)

	#excludesPixels from k-means thresholding 
	if excludePixels != None:
		Z[Z == excludePixels] = 255
	
	grayThresh,binary = cv2.threshold(image,absoluteThreshold,255,cv2.THRESH_BINARY)
	print(absoluteThreshold)

	contourImage, contours, hierarchy = getContours(binary, image)
	return binary, contours, contourImage, hierarchy

def iterativeThresholding(img, iterations, absoluteThresholds = None):
	image = img.copy()
	original_image = image.copy()
	newContourHistory = []
	allContourHistory = []
	newContourHierarchies = []
	allContourHierarchies = []

	for i in range(iterations):
		binaryImage, contours, contourImage, hierarchy = multiThresholding(image, absoluteThreshold = absoluteThresholds[i], excludePixels = 255)

		image[binaryImage == 0] = 255
		newContourHistory.append(contours)
		newContourHierarchies.append(hierarchy)

		binaryImage[image == 255] = 0
		contourImage, contours, hierarchy = getContours(binaryImage, original_image)

		allContourHistory.append(contours)
		allContourHierarchies.append(hierarchy)

	return binaryImage, contourImage, contours, newContourHistory, allContourHistory, newContourHierarchies, allContourHierarchies

def createFilterFrame(contours):
	features = []
	featureLabels = ['Outer Perimeter', 'Row LOC', 'Column LOC']

	for i in range(len(contours)):
		x,y,w,h = cv2.boundingRect(contours[i])
		centerLocation = [(y + h/2.0), (x + w/2.0)] #row, col
		perimeter = cv2.arcLength(contours[i], True)
		features.append([perimeter, centerLocation[0], centerLocation[1]])

	featureFrame = pd.DataFrame(features)
	featureFrame.columns = featureLabels
	return featureFrame

def createClassifierDataFrame(contours, image, hierarchy):
	classifierFrame = pd.DataFrame()
	classifierFrame['Contours'] = contours
	classifierFrame['Parent Hierarchy'] = hierarchy[0][0:, 3:4]
	return classifierFrame

def findNestedContours(df):
	dataFrame = df.copy()

	#find (nested) children and grandchildren contours of each contour
	children =  [ [] for i in range(len(dataFrame)) ]
	grandchildren = [ [] for i in range(len(dataFrame)) ]

	for i in range(len(dataFrame)):
		if dataFrame['Parent Hierarchy'][i] != -1:
			children[dataFrame['Parent Hierarchy'][i]].append(i)
	dataFrame['Children'] = children

	for i in range(len(dataFrame)):
		for j in dataFrame['Children'][i]:
			grandchildren[i] += dataFrame['Children'][j]
	dataFrame['Grandchildren'] = grandchildren


	#Establishing family tree for each contour--There should be a better way of doing this. I only need the generations, and a way to group indices into family trees besides making dictionaries.
	#should make this recursive to handle all possible nested cases, but should do for now (up to great-grandchildren; excludes possible great-great grandchildren)
	familyTree = [None] * len(dataFrame)
	generation = [None] * len(dataFrame)
	for i in range(len(dataFrame)):
		parentDict = {'Parent':i}
		if dataFrame['Parent Hierarchy'][i] == -1:
			assignments = [i]
			generation[i] = 0
			for j in dataFrame['Children'][i]:
				childDict = {}
				generation[j] = 1
				for k in dataFrame['Children'][j]:
					generation[k] = 2
					grandChildDict = {k: dataFrame['Children'][k]}
					childDict[k] = grandChildDict
					for l in dataFrame['Children'][k]:
						generation[l] = 3
						assignments.append(l)
					assignments.append(k)
				assignments.append(j)
				parentDict[j]=childDict

			if len(assignments)>1:
				for m in range(len(assignments)):
					familyTree[assignments[m]] = parentDict

		dataFrame['Generation'] = generation
		dataFrame['Family Tree'] = familyTree

	return dataFrame

def iterativeGrowthThresholder(image, absoluteThresholdList = [], kthreshList = [], plot = False):
	iterations = len(absoluteThresholdList)

	binaryImage, contourImage, contours, newContourHistory, allContourHistory, newContourHierarchies, allContourHierarchies= iterativeThresholding(image.copy(), iterations, absoluteThresholds = absoluteThresholdList)
	classifierDataFrame = createClassifierDataFrame(contours, image, allContourHierarchies[-1])
	nestedClassifierDataFrame = findNestedContours(classifierDataFrame)

	features = []
	featureLabels = ['Shape Area', 'Shape Mass', 'Outer Perimeter', 'Inner Perimeter', 'Row LOC', 'Column LOC', 'Row COM', 'Column COM','Moment of Inertia (Unweighted)', 'Radius of Gyration (Unweighted)', 'White Fringe Max', 'Pixel SD','Major Axis', 'Minor Axis', 'Ellipse Aspect Ratio', 'Ellipse Angle']

	w = 0
	skip = []
	while w < len(nestedClassifierDataFrame):
		if w in skip:
			w += 1
			continue

		if nestedClassifierDataFrame['Family Tree'][w] == None:
			shape = Shape(nestedClassifierDataFrame['Contours'][w], image, binaryImage)
			featureVector = shape.area, shape.cumulativeMass, shape.perimeter, 0, shape.centerLocation[0], shape.centerLocation[1], shape.COMLocation[0], shape.COMLocation[1], shape.moment_inertia, shape.radius_gyration, shape.maxWhiteFringe, shape.pixelSD, shape.majorAxis, shape.minorAxis, shape.ellipseAspectRatio, shape.ellipseAngle
			features.append(featureVector)
			w+=1 
		else:
			familyFrame = nestedClassifierDataFrame[nestedClassifierDataFrame['Family Tree'] == nestedClassifierDataFrame['Family Tree'][w]] 
			skip += list(familyFrame.index)
			shapes = NestedShapes(familyFrame.copy(), image, binaryImage)
			for i in range(len(shapes)):
				features.append(shapes[i])
			w += 1

	featureFrame = pd.DataFrame(features)
	featureFrame.columns = featureLabels

	comparisonFrame = createFilterFrame(newContourHistory[0])
	for i in range(len(newContourHistory)-1):
		nextFrame = createFilterFrame(newContourHistory[i+1])
		comparisonFrame = pd.concat([comparisonFrame, nextFrame], axis=0)

	featureFrame['Original Index'] = list(range(len(featureFrame)))
	featureFrame['Contours'] = allContourHistory[-1]

	isolatedContoursDataFrame = pd.merge(featureFrame, comparisonFrame[['Row LOC', 'Column LOC', 'Outer Perimeter']], on=['Row LOC', 'Column LOC', 'Outer Perimeter'], how='inner')
	isolatedContoursIndices = np.array(isolatedContoursDataFrame['Original Index'])
	isolatedContours = np.array(isolatedContoursDataFrame['Contours'])

	#Clean the data after finding nested contours
	featureFrame.drop(isolatedContoursIndices, inplace = True)
	holeContoursDF = featureFrame[(featureFrame['Shape Area'] == -1)]
	#whiteFringedContoursDF = featureFrame[(featureFrame['White Fringe Max'] == -1) | (featureFrame['Pixel SD'] / featureFrame['Shape Area'] >= 0.001)]

	if plot:
		image = np.array(image).astype('uint8') #adjustment
		clean_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		viewImage = cv2.drawContours(clean_image.copy(), np.array(featureFrame['Contours']), -1, (255,0,0), 1)
		viewImage = cv2.drawContours(viewImage, np.array(holeContoursDF['Contours']), -1, (0,0,255), 1)
		#viewImage = cv2.drawContours(clean_image.copy(), np.array(whiteFringedContoursDF['Contours']), -1, (255,0,0), 1)
		#viewImage = cv2.drawContours(viewImage, isolatedContours, -1, (255,255,0), 1)
		#plt.imshow(viewImage, cmap = 'gray')

	# Final Cleaning before returning droplet dataframes
	dropletFrame = featureFrame[featureFrame['Shape Area'] != -1] #removing holes from dropletFrame

	if plot:
		return dropletFrame, holeContoursDF, binaryImage, viewImage
	else:
		return dropletFrame, holeContoursDF, binaryImage

