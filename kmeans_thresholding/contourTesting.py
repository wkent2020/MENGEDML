#testing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#num = "0146"
#n = "146"
#img = cv2.imread('Honda/80bar//T2-X=0.00_Y=1.00__'+num+".tif",-1)
#img_rescaled = cv2.imread(direct + file, -1) #reads 16 bit without translating to 8 bit, if original file is in 16 bit
#cv2.imwrite("./norm_imgs/"+num+".png",img_rescaled)

def cropImage(image, cropTop=0, cropBottom = 0, cropLeft = 0, cropRight =0):
	"Crop pixels off the image"
	cropped_image = np.copy(image)
	cropped_image = cropped_image[cropTop:,cropLeft:]
	if cropBottom:
		cropped_image = cropped_image[:-cropBottom,]
	if cropRight:
		cropped_image = cropped_image[:,-cropRight]
	return cropped_image

direct = "Old_Groups_Code/norm_imgs/"
file = "300.tif"
img = cv2.imread(direct + file,-1)
img = cropImage(img)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)

plt.close('all')
#plt.imshow(img, cmap='gray')
#print(img)


def windowFrame(image, rows, columns, save = True):
	frames = []
	lenRows, lenColumns = image.shape
	for i in range(rows):
		for j in range(columns):
			picture = image[int(lenRows / rows) * i: int(lenRows / rows) * (i+1), int(lenColumns / columns) * j: int(lenColumns / columns) * (j+1)]
			frames.append(picture)

	if save == True:
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

def canny(img):
	edges = cv2.Canny(img,100,40)
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
	plt.show()
	return edges

def cannyClosing(img, kernelSize):
	edges = cv2.Canny(img, 100, 200)
	kernel = np.ones((kernelSize,kernelSize),np.uint8)
	closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	drawShapes(closing, img)
	return closing

def bilateralFilter(img):
	blur = cv2.bilateralFilter(img,9,75,75)
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(blur,cmap = 'gray')
	plt.title('Bilateral Filtered'), plt.xticks([]), plt.yticks([])
	plt.show()
	return blur

def medianFilter(img):
	median = cv2.medianBlur(img,5)
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(median,cmap = 'gray')
	plt.title('Median Filtered'), plt.xticks([]), plt.yticks([])
	plt.show()
	return median

def gaussianFilter(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(blur,cmap = 'gray')
	plt.title('Gaussian Filtered'), plt.xticks([]), plt.yticks([])
	plt.show()
	return blur

def averageBlur(img):
	blur = cv2.blur(img, (5,5))
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
	histogram(equ)
	blur = gaussianFilter(equ)
	histogram(blur)

def adaptiveHistogram(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(cl1,cmap = 'gray')
	plt.title('Adaptive Histogram'), plt.xticks([]), plt.yticks([])
	plt.show()
	histogram(cl1)

def laplace_sobel(img):
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	
	plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
	plt.title('Original'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
	plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
	plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
	plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	plt.show()

#frames = windowFrame(img, 5, 5)
#for i in range(len(frames)):
#	histogram(frames[i], show = False, save = 'frame' + str(i))

def contourArea(contours, image = None):
    #Return array of contour areas
    return np.array([cv2.contourArea(contour) for contour in contours])

def contourBoundary(contour):
	'''
	Save location of the shape in the context of the larger image
	'''

	x,y,w,h = cv2.boundingRect(contour)
	return np.array([y,y+h,x,x+w])

def cropContour(contour, image, border = 0):
	'''
	Crop shape from rest of image
	'''

	boundary = contourBoundary(contour) 
	# create a single channel pixel white image
	canvas = np.zeros(image.shape).astype(image.dtype) + 255
	fill = cv2.fillPoly(canvas, pts =[contour], color=0)
	#keep shape in grayscale, turn background white
	anti_fill = cv2.bitwise_or(image,fill)
	croppedContour = anti_fill[boundary[0]:boundary[1],\
					boundary[2]:boundary[3]]
	#also crop to slightly larger than boundary so shape isn't right at 
	#the edge of the image
	#this will be useful if we want to draw more contours on a shape after cropping it
	if border:
		borderedContour = anti_fill[boundary[0]-border:boundary[1]+border,\
					boundary[2]-border:boundary[3]+border]
		return borderedContour
	return croppedContour


def meanIntensity(contours, image):
    #Return mean intensity for each contour

	meanIntensities = []
	for contour in contours:
		croppedContour = cropContour(contour, image)
		mask = np.logical_not(np.logical_not(croppedContour -255)).astype('uint8')
		meanIntensities.append(cv2.mean(croppedContour, mask= mask)[0])
    
	return np.array(meanIntensities)


def drawShapes(image_binarized, image):
	'''
	Draw contours onto images
	'''
	contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	shapes_image = np.copy(image)

	#change back to RGB for easier visualization
	shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
#	plt.imshow(shapes_image) # uncomment these lines to plot in real time
#	plt.show()
	shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
	plt.imshow(shapes_image) 
	plt.show()

def kMeansHistogram(Z, label, kthresh, show = True, save = ''):
	'''
	Takes data for 1 dimensional kmeans and the resultant labels 
	and plots histogram of clusters
	'''
	fig, axs = plt.subplots(1, 1)
	for k in range(kthresh):
		cluster = Z[label == k]
		bins = 1+int(cluster.max() - cluster.min())
		axs.hist(cluster, bins, label = str(k))
	axs.legend()
	if show:
		plt.show()
	if len(save) != 0:
		os.system('mkdir savedHistograms')
		plt.savefig('savedHistograms/' + save)
		plt.close()

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
	
	drawShapes(proc, img)

def multiThresholding(image, kthresh, kthcenter = 0, plotHistogram = False):

	Z = image.reshape((-1,1))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #directly copied from opencv documentation
	ret,label,center = cv2.kmeans(Z,kthresh,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	if plotHistogram:
		kMeansHistogram(Z, label, kthresh)
	print(center)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
    # Puts something into single array
	res = center[label.flatten()]
    # Takes 1-D array and puts back into image format
	kthreshed = res.reshape((image.shape))
    # Lowest grayscale k-means center: aka some color
	if(kthcenter == 0):
		thresh_val = min(center)[0]
	else:
		thresh_val = np.sort(np.concatenate(center))[kthcenter]
	print("kmeans multithreshold value of "+str(thresh_val))            
	#now threshold the k-means clustered image to only keep the darkest cluster
	ret,proc = cv2.threshold(kthreshed,thresh_val,255,cv2.THRESH_BINARY) #binarizes
    # Binary image file
	print(proc)
    # Threshold value
	print(ret)
	#plt.imshow(proc,cmap = 'gray')
	#plt.show()
	#plt.imshow(img, cmap = 'gray')
	#plt.show()
	#canny(proc)

	
	drawShapes(proc, img)

	"""
	print("\n Performing multithresholding...")
	if data["kthresh"] != 0:
	    center1 = [] #darkest cluster
	    center2 = [] #next darkest cluster
	    total = 10
	    for loc in img_locs[:total]:
	        img = Image(loc)
	        min_center = min(img.center)[0]
	        min2_center = min(img.center[img.center != min(img.center)])
	        #print(img.center)
	        #print(min_center)
	        #print(min2_center)
	   
	        center1.append(min_center)
	        center2.append(min2_center)
	    center1 = np.median(center1)
	    center2 = np.median(center2)
	    #thresh_val = (center1 + center2)/2
	    thresh_val = center2
	    #print(thresh_val)
	    data["pixel_threshold"] = thresh_val
	    data["kthresh"] = 0         
	    with open('input.json','w') as f:
	        json.dump(data,f,indent=4)
	    print("final multithresholding value = %s \n" %(thresh_val)
	"""
#histogram(img)
multiThresholding(img, 3)

