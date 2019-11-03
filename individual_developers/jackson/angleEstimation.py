from sklearn import svm, preprocessing
import numpy as np 
import matplotlib.pyplot as plt 
from segmentedThresholding import getContours

img, binaryImage, image, contours = getContours()

split = 2/3
xLength = len(img)
yLength = len(img[0])

#Define a 2d grid that includes spray on left and right
cutoffRight = int(xLength*split)
cutoffLeft = int(xLength*(1-split))
xxRight = np.linspace(0,cutoffRight-1,cutoffRight)
xxLeft = np.linspace(cutoffLeft, xLength-1, xLength-cutoffLeft)
yy = np.linspace(0,yLength-1,yLength)
YYRight, XXRight = np.meshgrid(yy, xxRight)
xyRight = np.vstack([XXRight.ravel(), YYRight.ravel()]).T
YYLeft, XXLeft = np.meshgrid(yy, xxLeft)
xyLeft = np.vstack([XXLeft.ravel(), YYLeft.ravel()]).T
labelsLeft = binaryImage[cutoffLeft:].ravel()
labelsRight = binaryImage[:cutoffRight].ravel()


clfRight = svm.LinearSVC()
clfRight.fit(xyRight, labelsRight)
clfLeft = svm.LinearSVC()
clfLeft.fit(xyLeft, labelsLeft)