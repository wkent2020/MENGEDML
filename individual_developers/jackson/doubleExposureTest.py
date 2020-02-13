import cv2
import numpy as np
from scipy import fftpack
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

def fourierFilter(img, alpha):
    '''
    Apply the thickness finding Fourier algorithm 
    '''
    #Fourier frequency coordinates
    ky = fftpack.fftfreq(len(img))
    kx = fftpack.fftfreq(len(img[0]))
    kx, ky = np.meshgrid(kx, ky)
    #Compute their magnitudes
    fourierMagnitude = np.square(kx) + np.square(ky)

    #Perform Fourier transform
    img_ft = fftpack.fft2(img)
    #Divide by the fourier coordinate magnitudes
    transformedFT = img_ft/ (1+ alpha*fourierMagnitude)
    #Perform inverse Fourier transform
    transformedImage = fftpack.ifft2(transformedFT).real
    #Take the log, but do not divide by -mu
    thickness = np.log(transformedImage)

    return thickness


def meanIntensity(image):
    Z = np.concatenate(np.float32(image.reshape((-1,1))))
    return np.mean(Z)

def histogram(image, show = True, save = ''):
	Z = np.concatenate(np.float32(image.reshape((-1,1))))
	img_max = image.max()
	img_min = image.min()
	bins = img_max - img_min
	fig, axs = plt.subplots(1, 1)
	axs.hist(Z, bins)
	axs.set_xlim(15000,25000)
	if show:
		plt.show()
	if len(save) != 0:
		plt.savefig('individual_developers/jackson/norm_hist/' + save)
		plt.close()

def doubleFourier(img):
    '''
    Fourier transform the double exposure images 
    '''
    #Perform Fourier transform
    img_ft = fftpack.fft2(img)
    #Divide by the fourier coordinate magnitudes
    transformedFT = img_ft/ (1+ alpha*fourierMagnitude)
    #Perform inverse Fourier transform
    transformedImage = fftpack.ifft2(transformedFT).imaginary
    #Take the log, but do not divide by -mu
    thickness = np.log(transformedImage)

    return thickness

def normalize():
    '''
    Normalize all images using one reference point
    '''

    #Replace this code with our database framework

    location = 'individual_developers/jackson/doubleExposure/DE-9.55us-0'
    start = 3
    stop = 12
    output = 'individual_developers/jackson/doubleExposure/Fourier/'
    cropTop = 100
    background_frames = 6
    removeBackground = 1
    backgroundSection = [0,50,0,50]
    floatBoolean = 1
    fourier = 0
    alpha = (1.08475576*10**(-5))*(250*10**(-3)) / (808.572*10**(-6))

    '''
    with open('input.json') as f:
    data = json.load(f)

    #normalize one directory at a time
    location = data["img_dir"] + data["common_name"]

    #which images in the directory do we want to normalize?
    start = data["norm_imgs"][0]
    stop = data["norm_imgs"][1]

    #do we want to save the rescaled images
    write = 1 #1 to save rescaled images, 0 to not save them
    output = data["normalized_img_dir"]

    #number of frames to average when calculating the background,
    #you can change this for each set of spray images
    background_frames = data["nframes"]
    '''

    #calculate background
    imgs = []
    for n in range(start,background_frames+1):

        if n<10:
            num = "000"+str(n)
        else:
            num = "00"+str(n)

        img = cv2.imread(location+num+".tif", -1) #if we don't set -1 it will read as 8 bit

        #img = cropImage(img,cropTop=cropTop)

        if fourier:
            img = fourierFilter(img, alpha)

        imgs.append(img)

    bg = np.mean(imgs,axis=0)

    #subtract the background from each images,
    #then calculate the overall max/min pixel values.
    #These max/min values must be consistent across all sets of images we want to analyze!
    #Need to keep this in mind when normalizing multiple directories
    imgs_norm = []
    max_pixel = -1
    min_pixel = np.inf

    print("Subtracting background...")

    for n in range(start, stop+1):

        if n<10:
            num = "000"+str(n)
        elif n<100:
            num = "00"+str(n)
        else:
            num = "0"+str(n)
        
        img = cv2.imread(location+num+".tif",-1)
        
        #img = cropImage(img,cropTop=cropTop)
        
        if fourier:
            img = fourierFilter(img, alpha)

        if removeBackground:
            img_norm = img/bg
            #subtract background to bring background to 1
            #second_bg = np.mean(np.mean(img_nobg[backgroundSection[0]:backgroundSection[1], \
            #                    backgroundSection[2]:backgroundSection[3]]))
            #img_norm = img_nobg/second_bg
        else: 
            img_norm = img
        
        img_max = img_norm.max()
        img_min = img_norm.min()
        
        if img_max>max_pixel:
            max_pixel = img_max
        if img_min<min_pixel:
            min_pixel = img_min

        imgs_norm.append(img_norm)

    #now rescale each image to 8-bit range
    #pixel values will still be stored as floats
    imgs_rescaled = []
    img_intensities = []
    bg_intensities = []

    print("Rescaling images...")
    if floatBoolean:


        for img in imgs_norm:
            img_rescaled = 1.0*((img-min_pixel)/(max_pixel-min_pixel))
            #imgs_rescaled.append(img_rescaled.astype('32float'))

            img_ft = fftpack.fft2(img_rescaled)
            plt.close()
            fig, axes = plt.subplots(1, 3)
            axes[0].set_title("Original")
            axes[0].imshow(img_rescaled,cmap='gray')
            axes[1].set_title("Real Part")
            axes[1].imshow(img_ft.real,cmap='gray')
            axes[2].set_title("Imaginary Part")
            axes[2].imshow(img_ft.imag,cmap='gray')
            plt.tight_layout()
            plt.show()

        #write images -- this will automatically convert all values to uint8
        #for i in range(len(imgs_rescaled)):
        #    cv2.imwrite(output+str(i)+".tif",imgs_rescaled[i])
    else:
        #Convert back to 16-bit after the background removal
        #Does nothing if the background is not removed
        for img in imgs_norm:
            img_rescaled = (2**16)*((img-min_pixel)/(max_pixel-min_pixel))
            imgs_rescaled.append(img_rescaled.astype('uint16'))

        for i in range(len(imgs_rescaled)):
            img_intensities.append(meanIntensity(imgs_rescaled[i]).astype("32float")[0])
            #bg_intensities.append(meanIntensity(imgs_rescaled[i][backgroundSection[0]:backgroundSection[1], \
            #                    backgroundSection[2]:backgroundSection[3]]).astype("32float")[0])
            #if i == 120:
                #histogram(imgs_rescaled[i],False,str(i))
                #cv2.imwrite(output+str(i)+"bg"+str(backgroundSection[1])+".tif",imgs_rescaled[i])
                

        '''
        fig, axs = plt.subplots(1, 1)
        axs.plot(np.arange(0,399), np.array(bg_intensities))
        axs.set_title("Background Intensities : " + str(backgroundSection))
        axs.set_xlabel("mean: " + str(round(np.mean(np.array(bg_intensities)), 3)) + ", st dev: " + str(round(np.std(np.array(bg_intensities)), 3)))
        plt.show()
        print(str(np.mean(np.array(img_intensities))) + " , " + str(np.std(np.array(img_intensities))))
        '''        

normalize()