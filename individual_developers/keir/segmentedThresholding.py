import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

direct = 'Normalized_BackgroundRemoved/'
file = "120.tif"
img = cv2.imread(direct + file,-1)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)


class Shape(object):
    '''
    Shape class: creates a shape object for purposes of saving shape
      contours with a variety of parameters for clustering and analysis
    '''
    def __init__(self, contour):
        self.contour = contour
        self.label = False
        self.getArea()
        self.getApprox()
        self.getBoundary()
        self.getPeri()

    def getPeri(self):
        peri = cv2.arcLength(self.contour, True)
        self.peri = peri

    def getArea(self):
        '''
        Calculate area of shape
        '''
        self.area = cv2.contourArea(self.contour)

    def getApprox(self):
        '''
        Approximate shape as a polygon
        '''
        self.approx = cv2.approxPolyDP(self.contour,0.01*\
                      cv2.arcLength(self.contour,True),True)

    def getBoundary(self):
        '''
        Save location of the shape in the context of the larger image
        '''
        x,y,w,h = cv2.boundingRect(self.contour)
        self.h = h #height
        self.w = w #width
        self.boundary = [y,y+h,x,x+w]

    def crop(self,parent):
        '''
        Crop shape from rest of image
        '''
        #approximate shape of contour
        approx = cv2.approxPolyDP(self.contour, 0.02 * self.peri, True)
        # create a single channel pixel white image
        canvas = np.zeros(parent.shape).astype(parent.dtype) + 255
        fill = cv2.fillPoly(canvas, pts =[self.contour], color=0)
        #keep shape in grayscale, turn background white
        anti_fill = cv2.bitwise_or(parent,fill)
        self.cropped = anti_fill[self.boundary[0]:self.boundary[1],\
                      self.boundary[2]:self.boundary[3]]
        #also crop to slightly larger than boundary so shape isn't right at 
        #the edge of the image
        #this will be useful if we want to draw more contours on a shape after cropping it
        self.border = 2
        self.bordered = anti_fill[self.boundary[0]-\
                        self.border:self.boundary[1]+\
                        self.border,self.boundary[2]-\
                        self.border:self.boundary[3]+self.border]
        return(self.cropped)

    def pad(self,maxh,maxw):
        self.padded = np.pad(self.cropped,((0, maxh - self.h), \
                      (0, maxw - self.w)), 'constant', constant_values=0)

    def flatten(self):
        '''
        Flatten the 2-D array of the shape image into a 1-D array
        '''
        #when clustering, each shape is represented as a single row of values
        #only flatten AFTER padding
        self.flat = np.ndarray.flatten(self.padded)

    def setLabel(self,label):
        '''
        Assign the kmeans cluster label to the shape
        '''
        self.label = label


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


def segmentedThresholding(img, rows, columns, kthresh, filter = None, save = False, pixelThresh = False):
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

		#image_binarized = cv2.bitwise_not(image_binarized) #switch binary colors?

		contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		shapes_image = np.copy(image)
	
		#change back to RGB for easier visualization
		shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
		shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
		return shapes_image, contours

	allContours, contours = drawShapes(buildingColumn, img)
	return buildingColumn, allContours, contours

def imagePad(image, top, bottom, left, right, greyscale = 255):
	rows, columns = image.shape
	padTopBottom = np.zeros(columns) + greyscale
	padRight = np.zeros(rows) + greyscale
	for i in range(top):
		image = np.vstack((padTopBottom, image))
	for i in range(bottom):
		image = np.vstack((image, padTopBottom))
	padRight = np.array([np.zeros(rows+top+bottom) + greyscale])
	for i in range(left):
		image = np.insert(image, 0, greyscale, axis=1)
	for i in range(right):
		image = np.concatenate((image, padRight.T), axis=1)
	return image

def getContours():
	binaryImage, image, contours = segmentedThresholding(img, 1, 1, 3, filter = None, pixelThresh = False, save = False)
	return img, binaryImage, image, contours
"""
segmentedThresholding Function:

Arguments:
	8bit Image File
	# of Rows to be windowframed
	# of Columns to be windowframed
	# of k-means centers
	Filters to be sequentially applied, with the names of the filtering functions stored in a list (in order of filters to be applied). If None, no filter is used. 
	pixelThresh: If True, thresholds based on the average greyscale value of the two darkest k-means. If False, thresholds by condidering the pixels assigned to the darkest k-means cluster.

Returns: binary image, image with contours
"""

def main():
	#image = canny(img)
	binaryImage, image, contours = segmentedThresholding(img, 1, 1, 3, filter = None, pixelThresh = False, save = False)
	#histogram(contourImage)
	#plt.imshow(image, cmap = 'gray')
	#plt.title('Histogram Equalization + Median Filter')
	#plt.show()
	plt.imshow(image, cmap = 'gray')
	plt.show()
	
	plt.imshow(binaryImage[8:21, 241:288], cmap = 'gray')#image[8:21, 241:288]
	plt.show()
	
	for i in range(len(contours)):
		shape = Shape(contours[i])
		if shape.boundary[0] == 8 and shape.boundary[1] == 21:#.area >100 and shape.area < 110: #>200
			choose = i
			break
	
	shape = Shape(contours[choose])
	shapeCrop = shape.crop(img)
	arrayShape  = shapeCrop.shape
	r1, r2, c1, c2 = shape.boundary
	print(shape.boundary)
	plt.imshow(image[r1:r2, c1:c2])#image[8:21, 241:288]
	plt.show()
	
	#Turning all background to black (0) for density estimation
	shapeCrop = shapeCrop.flatten()
	backgroundCount = 0
	pixelCount = len(shapeCrop)
	for i in range(pixelCount):
		if shapeCrop[i] == 255:
			backgroundCount += 1
			shapeCrop[i] = 0
	
	print(shapeCrop)
	print('Area:',shape.area)
	print('Perimeter:', shape.peri)
	
	summing = 0
	number = 0
	for i in range(len(shapeCrop)):
		if shapeCrop[i] != 0:
			summing += shapeCrop[i]
			number += 1
	print('Number of Pixels:',number)
	averageDensity = np.sum(shapeCrop) / (shape.area)
	print('Density using shape.area',averageDensity)
	print('Density using pixel number',summing/number)
	
	#turning all background back to white for visualization
	for i in range(pixelCount):
		if shapeCrop[i] == 0:
			shapeCrop[i] = 255
	
	shapeCrop = shapeCrop.reshape(arrayShape)
	shapeCrop = imagePad(shapeCrop, 3, 3, 3, 3, greyscale = 255)
	plt.imshow(shapeCrop, cmap = 'gray')
	plt.show()
	#histogram(shapeCrop)
	kernel = np.ones((3,3), np.uint8) 
	#img_erosion = cv2.erode(shapeCrop, kernel, iterations=1) 
	img_dilation= cv2.erode(shapeCrop, kernel, iterations=1)
	plt.imshow(img_dilation, cmap = 'gray')
	plt.show()
	#for i in range(len(contours)):
	#	print(len(i))
	
if __name__=='__main__':
	main()