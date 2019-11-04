import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Shape import *
from segmentedThresholding import getContours, histogramEqualization, gaussianFilter, averageBlur

def createTestDataFrame(contours, image, location = False):
	features = []
	if location:
		featureLabels = ['Shape Area', 'Shape Density', 'Shape Perimeter', 'Row Location', 'Column Location']
	else:
		featureLabels = ['Shape Area', 'Shape Density', 'Shape Perimeter']
	for i in range(len(contours)):
		shape = Shape(contours[i], image)
		if location:
			featureVector = shape.area, shape.meanDensity, shape.perimeter, shape.centerLocation[0], shape.centerLocation[1]
		else:
			featureVector = shape.area, shape.meanDensity, shape.perimeter
		features.append(featureVector)
	featureFrame = pd.DataFrame(features)
	featureFrame.columns = featureLabels
	return featureFrame

def createFalseDataFrame(contours, image, classify, location = False):
	features = []
	if location:
		featureLabels = ['Shape Area', 'Shape Density', 'Shape Perimeter', 'Row Location', 'Column Location', 'Classification',]
	else:
		featureLabels = ['Shape Area', 'Shape Density', 'Shape Perimeter', 'Classification']
	for i in range(len(contours)):
		shape = Shape(contours[i], image)
		if shape.centerLocation[1] > 50 and shape.centerLocation[1] < 450:
			continue
		if location:
			featureVector = shape.area, shape.meanDensity, shape.perimeter, shape.centerLocation[0], shape.centerLocation[1], classify
		else:
			featureVector = shape.area, shape.meanDensity, shape.perimeter, classify
		features.append(featureVector)
	featureFrame = pd.DataFrame(features)
	featureFrame.columns = featureLabels
	return featureFrame

def createTrueDataFrame(contours, image, classify, location = False):
	features = []
	if location:
		featureLabels = ['Shape Area', 'Shape Density', 'Shape Perimeter', 'Row Location', 'Column Location', 'Classification']
	else:
		featureLabels = ['Shape Area', 'Shape Density', 'Shape Perimeter', 'Classification']
	for i in range(len(contours)):
		shape = Shape(contours[i], image)
		if shape.area > 2000 or shape.area < 3:
			continue
		if location:
			featureVector = shape.area, shape.meanDensity, shape.perimeter, shape.centerLocation[0], shape.centerLocation[1], classify
		else:
			featureVector = shape.area, shape.meanDensity, shape.perimeter, classify
		features.append(featureVector)
	featureFrame = pd.DataFrame(features)
	featureFrame.columns = featureLabels
	return featureFrame

def trainClassifier(featureFrame, classification):
	forest = RandomForestClassifier(n_estimators = 100)
	target = np.array(classification)
	featureFrameArray = np.array(featureFrame)

	train_descriptors, test_descriptors, train_target, test_target = train_test_split(featureFrameArray, target, test_size = 0.25)
	forest.fit(train_descriptors, train_target)
	
	model_predictions = forest.predict(test_descriptors)

	model_error = np.sum(np.abs(np.array(model_predictions) - np.array(test_target))) / len(test_target) #change to classification case
	#model_error = forest.score(test_descriptors, test_target)
	return forest, model_predictions, model_error, forest.feature_importances_

def applyClassifier(model, featureFrame):
	classification = model.predict(featureFrame)
	return classification



#Creates Data Frame containing true droplets
image, binaryImage, contourImage, contours = getContours(1, 1, filters = None)
trueFrame = createTrueDataFrame(contours, image, 1.0)
trueClassify= trueFrame['Classification']
trueFrame.drop('Classification', 1, inplace = True)

#Creates Data Frame containing false droplets
image, binaryImage, contourImage, contours = getContours(3, 3, filters = [histogramEqualization])
falseFrame = createFalseDataFrame(contours, image, 0.0)
falseClassify = falseFrame['Classification']
falseFrame.drop('Classification', 1, inplace = True)

#Combining data frames
totalFrame = pd.concat([trueFrame, falseFrame], axis=0)
totalClassify = pd.concat([trueClassify, falseClassify], axis=0)

#Train Random Forest Algorithm
forestModel, testPredict, testError, featureWeights = trainClassifier(totalFrame, totalClassify)
print((1.0 - testError) * 100.0, featureWeights) #prints accuracy and feature importance of test set

#Apply Random Forest Algorithm to classify unlabeled data
image, binaryImage, contourImage, contours = getContours(3, 3, filters = [histogramEqualization])
unlabeledFeatures = createTestDataFrame(contours, image)
predictedClasses = applyClassifier(forestModel, unlabeledFeatures)

plt.imshow(contourImage)
plt.show()
classifiedContours=[]
for i in range(len(predictedClasses)):
	if predictedClasses[i] == 1.0:
		classifiedContours.append(contours[i])
classifiedContours = np.array(classifiedContours) 
classifiedImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
classifiedImage = cv2.drawContours(classifiedImage, classifiedContours, -1, (255,0,0), 1)
plt.imshow(classifiedImage)
plt.show()

