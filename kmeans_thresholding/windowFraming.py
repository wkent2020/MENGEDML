import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#num = "0146"
#n = "146"
#img = cv2.imread('Honda/80bar//T2-X=0.00_Y=1.00__'+num+".tif",-1)
#img_rescaled = cv2.imread(direct + file, -1) #reads 16 bit without translating to 8 bit, if original file is in 16 bit
#cv2.imwrite("./norm_imgs/"+num+".png",img_rescaled)


direct = 'Normalized_BackgroundRemoved/'
file = "350.tif"
img = cv2.imread(direct + file,-1)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)

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

histogram(img)
frames = windowFrame(img, 5, 5)
for i in range(len(frames)):
	histogram(frames[i], show = False, save = 'frame' + str(i))

