from sklearn import svm, preprocessing
import numpy as np 
import matplotlib.pyplot as plt 
from segmentedThresholding import getContours

img, binaryImage, image, contours = getContours()

split = 1/2
yLength = len(img)
xLength = len(img[0])
crop = 30

#Define a 2d grid that includes spray on left and right
cutoffRight = int(xLength*split)
cutoffLeft = int(xLength*(1-split))
xxRight = np.linspace(0,cutoffRight-1,cutoffRight)
xxLeft = np.linspace(cutoffLeft, xLength-1, xLength-cutoffLeft)
yy = np.linspace(crop,yLength-1,yLength-crop)
YYRight, XXRight = np.meshgrid(yy, xxRight)
xyRight = np.vstack([XXRight.ravel(), YYRight.ravel()]).T
XXLeft, YYLeft = np.meshgrid(xxLeft, yy)
xyLeft = np.vstack([XXLeft.ravel(), YYLeft.ravel()]).T
#Make an array with where each entry indicates 
# whether that coordinate is spray or background
labelsLeft = (binaryImage[crop:,cutoffLeft:].ravel()/255).astype('uint16')
labelsRight = binaryImage[crop:,:cutoffRight].ravel()

classWeight = {0: 10}
maxIter = 100000

'''
clfRight = svm.LinearSVC(random_state=0, tol=1e-5)
clfRight.fit(xyRight, labelsRight)
'''
#Fit the support vector machine with higher 
# weight on spray than background
clfLeft = svm.LinearSVC(class_weight=classWeight, max_iter=maxIter)
clfLeft.fit(xyLeft, labelsLeft)

#Plot the results
plt.scatter(xyLeft[:, 0], xyLeft[:, 1], c=labelsLeft, cmap='gray', s=1)
ax = plt.gca()
Z = clfLeft.predict(np.c_[XXLeft.ravel(),YYLeft.ravel()]).reshape(XXLeft.shape)
ax.contourf(XXLeft, YYLeft, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.show()