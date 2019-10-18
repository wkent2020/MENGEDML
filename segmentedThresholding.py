import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

direct = 'Normalized_BackgroundRemoved/'
file = "121.tif"
img = cv2.imread(direct + file,-1)
img = cropImage(img)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)

def cropImage(image, cropTop=0, cropBottom = 0, cropLeft = 0, cropRight =0):
	"Crop pixels off the image"
	cropped_image = np.copy(image)
	return cropped_image[cropTop:-cropBottom,cropLeft:-cropRight]


def windowFrame(image, rows, columns, save = True):
	frames = []
	lenRows, lenColumns = image.shape
	for i in range(rows):
		for j in range(columns):
			picture = image[int(lenRows / rows) * i: int(lenRows / rows) * (i+1), int(lenColumns / columns) * j: int(lenColumns / columns) * (j+1)]
			frames.append(picture)
	if save:
		os.system('mkdir windowFrames_'+file[0:-4])
		for i in range(len(frames)):
			cv2.imwrite('windowFrames_' +file[0:-4] + "/frame"+str(i)+".tif",frames[i])
	return frames

def histogram(image, show = True, save = ''):
	Z = np.concatenate(np.float32(image.reshape((-1,1))))
	img_max = image.max()
	img_min = image.min()
	bins = img_max - img_min
	fig, axs = plt.subplots(1, 1)
	axs.hist(Z, bins)
	#axs.set_xlim(0,255)
	if show:
		plt.show()
	if len(save) != 0:
		os.system('mkdir savedHistograms')
		plt.savefig('savedHistograms/' + save)
		plt.close()

def canny(img, plot = False):
	edges = cv2.Canny(img,100,40)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
		plt.show()
	return edges

def bilateralFilter(img, plot = False):
	blur = cv2.bilateralFilter(img,9,75,75)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(blur,cmap = 'gray')
		plt.title('Bilateral Filtered'), plt.xticks([]), plt.yticks([])
		plt.show()
	return blur

def medianFilter(img, plot = False):
	median = cv2.medianBlur(img,5)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(median,cmap = 'gray')
		plt.title('Median Filtered'), plt.xticks([]), plt.yticks([])
		plt.show()
	return median

def gaussianFilter(img, plot = False):
	blur = cv2.GaussianBlur(img,(5,5),0)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(blur,cmap = 'gray')
		plt.title('Gaussian Filtered'), plt.xticks([]), plt.yticks([])
		plt.show()
	return blur

def averageBlur(img, plot = False):
	blur = cv2.blur(img, (5,5))
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(blur,cmap = 'gray')
		plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
		plt.show()
	return blur

#blur = gaussianFilter(img)
#histogram(blur)

def histogramEqualization(img):
	equ = cv2.equalizeHist(img)
	return equ

def adaptiveHistogram(img, plot = False):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	if plot:
		plt.subplot(121),plt.imshow(img,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(cl1,cmap = 'gray')
		plt.title('Adaptive Histogram'), plt.xticks([]), plt.yticks([])
		plt.show()
	return cl1

def laplace(img, plot = False):
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	if plot:
		plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
		plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
		plt.show()
	return laplacian

def sobelx(img, plot = False):
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	if plot:
		plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
		plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	return sobelx

def sobely(img, plot = False):
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	if plot:
		plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(2,2,2),plt.imshow(sobely,cmap = 'gray')
		plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	return sobely

def multiThresholding(image, kthresh, pixelThresh = False):
	Z = image.reshape((-1,1))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #directly copied from opencv documentation
	ret,label,center=cv2.kmeans(Z,kthresh,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS) #sum of squared errors, labels, greyscale centers
	print(center.flatten())
	#print(ret)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	sortedCenters = sorted(center.flatten())
	res = center[label.flatten()]
	kthreshed = res.reshape((image.shape))

	#now threshold the k-means clustered image to only keep the darkest cluster
	if pixelThresh:
		thresh_val = (sortedCenters[1] + sortedCenters[0])/2
		#thresh_val = sortedCenters[0]
		ret,proc = cv2.threshold(image,thresh_val,255,cv2.THRESH_BINARY)

	else:
		thresh_val = sortedCenters[0]
		ret, proc = cv2.threshold(kthreshed,thresh_val,255,cv2.THRESH_BINARY) #greyscale threshold, binary image

	frame_contours = []	
	def drawShapes(image_binarized, image):
		'''
		Draw contours onto images
		'''
		contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		shapes_image = np.copy(image)
		#change back to RGB for easier visualization
		shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
		shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
		return shapes_image

	contourImage = drawShapes(proc, image)
	return proc, contourImage


def segmentedThresholding(img, rows, columns, kthresh, filter = None, save = True, pixelThresh = False):
	frames = windowFrame(img, rows, columns, save)
	frames_contours = []
	for i in range(len(frames)):
		filteredFrame = frames[i]
		if filter != None:
			for fil in filter:
				filteredFrame = fil(filteredFrame)
		frame_shapes, contourImage = multiThresholding(filteredFrame, kthresh, pixelThresh = pixelThresh)
		frames_contours.append(frame_shapes)
		#histogram(framesi[], show = False, save = 'frame' + str(i))
	
	if save:
		os.system('mkdir frameContours_'+file[0:-4])
		for i in range(len(frames_contours)):
			cv2.imwrite('frameContours_' +file[0:-4] + "/frame"+str(i)+".tif",frames_contours[i])
	
	#recombines segmented binary files into single binary image
	i = 0
	while i < (len(frames_contours)):
		for j in range(columns):
			if j == 0:
				buildingRow = frames_contours[i]
				continue
			if (j+1) <= columns:
				buildingRow = np.hstack((buildingRow, frames_contours[i+j]))

		if i ==0:
			buildingColumn = buildingRow.copy()
		else:
			buildingColumn = np.vstack((buildingColumn, buildingRow))

		i += columns

	
	def drawShapes(image_binarized, image):
		'''
		Draw contours onto images
		'''
		contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		shapes_image = np.copy(image)
	
		#change back to RGB for easier visualization
		shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
		shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
		return shapes_image

	allContours = drawShapes(buildingColumn, img)
	return buildingColumn, allContours

#arguments: loaded 8bit image file, # rows, # columns, # of k-means clusters, filters to be applied (names of functions, in python list. If no filters to be applied, ),
"""
SegmentingThresholding Function:
Arguments:
	8bit Image File
	# of Rows to be windowframed
	# of Columns to be windowframed
	# of k-means centers
	Filters to be sequentially applied, with the names of the filtering functions stored in a list (in order of filters to be applied). If None, no filter is used. 
	pixelThresh: If True, thresholds based on the average greyscale value of the two darkest k-means. If False, thresholds by condidering the pixels assigned to the darkest k-means cluster.

Returns binary image, image with contours
"""

binaryImage, contourImage = segmentedThresholding(img, 5, 5, 3, filter = None, pixelThresh = False)
plt.imshow(contourImage)
plt.show()


