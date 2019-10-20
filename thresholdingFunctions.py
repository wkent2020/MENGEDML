import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cropImage(image, cropTop=0, cropBottom = 0, cropLeft = 0, cropRight =0):
	"Crop pixels off the image"
	cropped_image = np.copy(image)
	cropped_image = cropped_image[cropTop:,cropLeft:]
	if cropBottom:
		cropped_image = cropped_image[:-cropBottom,]
	if cropRight:
		cropped_image = cropped_image[:,-cropRight]
	return cropped_image

def windowFrame(image, rows, columns, save = True, file='none'):
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

def windowFrame2(image, rows, columns, rmdiv, cmdiv, save = True, file='none'):
	frames = []
	lenRows, lenColumns = image.shape
	for i in range(rows):
		for j in range(columns):
			picture = image[int(lenRows / rows) * i: int(lenRows / rows) * (i+1), int(lenColumns / columns) * j: int(lenColumns / columns) * (j+1)]
			frames.append(picture)
	if columns % 2 == 1:
		midframe = frames[int(columns/2)]
		mlenRows, mlenColumns = midframe.shape
		midframes = []
		for i in range(cmdiv):
			newframe = midframe[0 : int(mlenRows / rmdiv), int(mlenColumns / cmdiv)*i: int(mlenColumns / cmdiv)*(i+1)]
			midframes.append(newframe)
		del frames[int(columns/2)]
		for i in range(cmdiv):
			frames.insert(int(columns/2)+i, midframes[i])
		
	# Have not tested function for even number of columns/rows
	if columns % 2 == 0:
		midframe = frames[int(columns/2)+1]
		mlenRows, mlenColumns = image.shape(midframe)
		leftframe = midframe[0: int(mlenRows / rmdiv), 0: int(mlenColumns / cmdiv)]
		rightframe = midframe[int(mlenRows / rmdiv):mlenRows , int(mlenColumns / 2):mlenRows]
    
	if save:
		os.system('mkdir windowFrames_'+file[0:-4])
		for i in range(len(frames)):
			cv2.imwrite('windowFrames_' +file[0:-4] + "/frame"+str(i)+".tif",frames[i])
	return frames

def windowFrame3(image, rows, columns, factor, save = True, file='none'):
	frames = []
	lenRows, lenColumns = image.shape
	
	colfac = 0
	rowfac = 0
	columnvals = [0]
	rowvals = [0]
	cit = int((columns/2))
	rit = int((rows/2))
	if columns%2 == 0:
		for i in range(cit):
			colfac += factor**i
		
		# Computs the width of first column
		column1 = lenColumns/(2*colfac)
		colend = 0
		
		# Iteratively computes addresses of the right of the column
		for i in range(cit):
			colend += column1*(factor**i)
			columnvals.append(colend)
			
		for i in range(cit):
			colend += (columnvals[cit-i]-columnvals[cit-i-1])
			columnvals.append(colend)
	else:
		for i in range(cit+1):
			if i == cit:
				colfac += 0.5*(factor**i)
			else:
				colfac += factor**i
#		print("Colfac is %d" % colfac)
		# Computs the width of first column
		column1 = lenColumns/(2*colfac)
		colend = 0
		
		# Iteratively computes addresses of the right of the column
		for i in range(cit+1):
			colend += column1*(factor**i)
			columnvals.append(colend)
			
		for i in range(cit):
			colend += (columnvals[cit-i]-columnvals[cit-i-1])
			columnvals.append(colend)
		
	if rows%2 == 0:	
		for i in range(rit):
			rowfac += factor**i
			
		row1 = lenRows/(2*rowfac)
		rowend = 0
		
		# Iteratively computes addresses of the bottom of the rows
		for i in range(rit):
			rowend += row1*(factor**i)
			rowvals.append(rowend)
			
		for i in range(rit):
			rowend += (rowvals[rit-i]-columnvals[rit-i-1])
			rowvals.append(rowend)
	elif rows == 1:
		rowvals.append(lenRows)
	else:
		for i in range(rit+1):
			if i == rit:
				rowfac += 0.5*(factor**i)
			else:
				rowfac += factor**i
#		print("Colfac is %d" % rowfac)
		# Computs the width of first column
		row1 = lenRows/(2*rowfac)
		rowend = 0
		
		# Iteratively computes addresses of the right of the column
		for i in range(rit+1):
			rowend += row1*(factor**i)
			rowvals.append(rowend)
			
		for i in range(cit):
			rowend += (rowvals[rit-i]-rowvals[rit-i-1])
			rowvals.append(rowend)
	

		
#	outputcols  = "The number of columns: %i"  % (len(columnvals)-1)
#	outputrows  = "The number of rows: %i"  % (len(rowvals)-1)
#	print(outputcols)
#	print(outputrows)
#	print(rowvals)
#	print(columnvals)
	
	for i in range(len(rowvals)-1):
		for j in range(len(columnvals)-1):
#			outputcols  = "The left column val: %i"  % int(columnvals[j]) 
#			outputrows  = "The row val: %i"  % int(rowvals[i])
#			print(outputcols)
#			print(outputrows)
			picture = image[int(rowvals[i]): int(rowvals[i+1]), int(columnvals[j]) : int(columnvals[j+1])]
			frames.append(picture)
    
	if save:
		os.system('mkdir windowFrames_'+file[0:-4])
		for i in range(len(frames)):
			cv2.imwrite('windowFrames_' +file[0:-4] + "/frame"+str(i)+".tif",frames[i])
#	print("The number of frames is %i" % len(frames))
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

def adaptiveThresholding(image, thresholdType = 1, blockSize = 11, subtract = 2):
	'''
	Applies adaptive thresholding to image with either mean or Gaussian thresholding
	thresholdType of true gives adaptive mean thresholding and false gives adaptive Gaussian thresholding
	blockSize sets the size of the neighborhood, and subtract reduces the threshold by the given amount
	'''
	if thresholdType:
		thresh = cv2.ADAPTIVE_THRESH_MEAN_C
	else:
		thresh = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

	proc = cv2.adaptiveThreshold(image, 255, thresh, \
		cv2.THRESH_BINARY, blockSize, subtract)

	return proc

def drawShapes(image_binarized, image):
	'''
	Draw contours onto images
	'''
	contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	shapes_image = np.copy(image)
	
	print("There are %d contours" % len(contours))
	#change back to RGB for easier visualization
	shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
	shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
	return shapes_image