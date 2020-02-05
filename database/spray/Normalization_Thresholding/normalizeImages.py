import cv2
import numpy as np
import matplotlib.pyplot as plt

def cropImage(image, cropTop=0, cropBottom = 0, cropLeft = 0, cropRight =0):
	"Crop pixels off the image"
	cropped_image = np.copy(image)
	cropped_image = cropped_image[cropTop:,cropLeft:]
	if cropBottom:
		cropped_image = cropped_image[:-cropBottom,]
	if cropRight:
		cropped_image = cropped_image[:,0:-cropRight]
	return cropped_image

def divideAverageBackground(images, start, end):
	mean_image = np.array(images[start]).copy()
	mean_image[mean_image > 0] = 0
	for i in range(start, end):
		mean_image += images[i]
	
	mean_image  = mean_image / (float(end-start))

	#mean_image = np.array(np.round(mean_image),dtype = int) #integer

	normalized_images = []
	for i in images:
		normalized_images.append(i/mean_image)
	return normalized_images

def rescale_minmax(images, newmin, newmax):
	pixel_min = np.inf
	pixel_max = 0.0

	for i in range(len(images)):
		if np.min(images[i]) < pixel_min:
			pixel_min = np.min(images[i])
		if np.max(images[i]) > pixel_max:
			pixel_max = np.max(images[i])

	rescaled_images = []
	for i in images:
		rescaled_i = ((i-pixel_min)*newmax/(pixel_max-pixel_min)) + newmin
		rescaled_images.append(rescaled_i)
	return rescaled_images

def round_nearestInt(images):
	rounded_images = []
	for i in images:
		round_image = np.array(np.round(i),dtype = int) #integer
		rounded_images.append(round_image)
	return rounded_images

def divideIndivuallyByMean(images):
	norm_images = []
	for i in images:
		mean = np.mean(i)
		norm_images.append(i/mean)
	return norm_images

def divideIndividuallyByBackgroundSelection(images, backgroundSection):
	norm_images = []
	for i in images:
		backgroundMean = np.mean(i[backgroundSection[0]:backgroundSection[1], backgroundSection[2]:backgroundSection[3]])
		norm_images.append(i/backgroundMean)
	return norm_images

def Normalize(images, scaleFactor, background_images_indices, backgroundSection, mode = 'DBB'):
	images = divideAverageBackground(images, background_images_indices[0], background_images_indices[1]) #single background
	
	if mode == 'SB':
		images = rescale_minmax(images, 0, scaleFactor)
		return images
	else:
		if mode == 'DBB':
			images = divideIndividuallyByBackgroundSelection(images, backgroundSection) #double background, divide by background section
		elif mode == 'DBM':
			images = divideIndivuallyByMean(images) #double background, divide by image mean
		
		images = rescale_minmax(images, 0, scaleFactor)
		return images


