import numpy as np
import matplotlib.pyplot as plt
import cv2
#import cousin-directory libraries

def cropImage(image, cropTop=0, cropBottom = 0, cropLeft = 0, cropRight =0):
	"Crop pixels off the image"
	cropped_image = np.copy(image)
	cropped_image = cropped_image[cropTop:,cropLeft:]
	if cropBottom:
		cropped_image = cropped_image[:-cropBottom,]
	if cropRight:
		cropped_image = cropped_image[:,-cropRight]
	return cropped_image

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

backgroundImage = 'test_images/120.tif'
backgroundImage1 = 'individual_developers/jackson/120.tif'

img = cv2.imread(backgroundImage,-1)
img1 = cv2.imread(backgroundImage1,-1)

print(img[0,0])
print(img1[0,0])
print("")
print(img.shape)
print(img1.shape)
#backgroundImage = '../../test_images/20.tif' #pure background
'''
backgroundImage = processImage(backgroundImage, cropLeft = 0, cropTop =0, cropBottom = 200, cropRight =420)

plt.imshow(backgroundImage, cmap ='gray')
plt.show()
print(np.mean(backgroundImage))
histogram(backgroundImage)
'''

