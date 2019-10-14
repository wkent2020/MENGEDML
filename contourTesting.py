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


direct = "Old_Groups_Code/norm_imgs/"
file = "300.tif"
img = cv2.imread(direct + file,-1)
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

def multiThresholding(image, kthresh):

	Z = image.reshape((-1,1))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #directly copied from opencv documentation
	ret,label,center = cv2.kmeans(Z,kthresh,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	print(center)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
    # Puts something into single array
	res = center[label.flatten()]
    # Takes 1-D array and puts back into image format
	kthreshed = res.reshape((image.shape))
    # Lowest grayscale k-means center: aka some color
	thresh_val = min(center)[0]
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

	def drawShapes(image_binarized, image):
		'''
		Draw contours onto images
		'''
		contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		shapes_image = np.copy(image)

		#change back to RGB for easier visualization
		shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
#		plt.imshow(shapes_image) # uncomment these lines to plot in real time
#		plt.show()
		shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)
		plt.imshow(shapes_image) 
		plt.show()
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

