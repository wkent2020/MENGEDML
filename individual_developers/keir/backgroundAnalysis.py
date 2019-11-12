import numpy as np
import matplotlib.pyplot as plt
import cv2
#import cousin-directory libraries
import sys
sys.path.append("../..")
from auxiliary_functions.contourFunctions import  *
from auxiliary_functions.thresholdingFunctions import *
from kmeans_thresholding.segmentedThresholding import *
from kmeans_thresholding.iterativeThresholding import *
from shape_information.Shape import *
from image_processing.image_filters import *
from image_processing.cropping.cropImage import *

def processImage(imageDirectory, cropTop = 0, cropLeft = 0, cropRight = 0, cropBottom = 0):
	img = cv2.imread(imageDirectory,-1)
	img = cropImage(img, cropTop = 100)
	img_max = img.max()
	img_min = img.min()
	img_rescaled = img
	#img_rescaled = 255*((img-img_min)/(img_max-img_min)) #don't use
	image = np.array(img_rescaled, dtype = np.uint8)
	image = cropImage(image, cropLeft = cropLeft, cropRight = cropRight, cropBottom = cropBottom)
	return image

def histogram(image):
	Z = np.concatenate(np.float32(image.reshape((-1,1))))
	img_max = image.max()
	img_min = image.min()
	bins = img_max - img_min
	fig, axs = plt.subplots(1, 1)
	axs.hist(Z, bins)
	#axs.set_xlim(0,255)
	plt.show()

backgroundImage = '../../test_images/125.tif'
#backgroundImage = '../../test_images/20.tif' #pure background

backgroundImage = processImage(backgroundImage, cropLeft = 0, cropTop =0, cropBottom = 200, cropRight =420)

plt.imshow(backgroundImage, cmap ='gray')
plt.show()
print(np.mean(backgroundImage))
histogram(backgroundImage)


