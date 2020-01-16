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

def normalize():
    '''
    Normalize all images using one reference point
    '''

    #Replace this code with our database framework

    location = 'Old_Groups_Code/Honda/80bar/T2-X=0.00_Y=1.00__'
    start = 1
    stop = 399
    output = 'individual_developers/jackson/norm_Img/16BitFourierSamples/'
    cropTop = 100
    background_frames = 6
    removeBackground = 1
    backgroundSection = [0,100,412,512]
    floatBoolean = 0
    fourier = 1
    energies = [7000,8096.56,10071.7246,30000]
    alphas = [(1.08475576*10**(-5))*(250*10**(-3)) / (808.572*10**(-6)), (8.10269285*10**(-6))*(250*10**(-3)) / (1265.81*10**(-6)), (5.2321966*10**(-6))*(250*10**(-3)) / (2445.54*10**(-6)), (5.8842113*10**(-7))*(250*10**(-3)) / (18975*10**(-6))]
    energy = energies[3]
    alpha = alphas[3] 

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

        img = cropImage(img,cropTop=cropTop)

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

        img = cropImage(img,cropTop=cropTop)
        
        if fourier:
            img = fourierFilter(img, alpha)

        if removeBackground:
            img_nobg = img/bg
            #subtract background to bring background to 1
            second_bg = np.mean(np.mean(img_nobg[backgroundSection[0]:backgroundSection[1], \
                                backgroundSection[2]:backgroundSection[3]]))
            img_norm = img_nobg/second_bg
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

    print("Rescaling images...")
    if floatBoolean:


        for img in imgs_norm:
            img_rescaled = 1.0*((img-min_pixel)/(max_pixel-min_pixel))
            imgs_rescaled.append(img_rescaled.astype('32float'))

        #write images -- this will automatically convert all values to uint8
        for i in range(len(imgs_rescaled)):
            cv2.imwrite(output+str(int(energy))+str(120)+".tif",imgs_rescaled[120])
    else:
        #Convert back to 16-bit after the background removal
        #Does nothing if the background is not removed
        for img in imgs_norm:
            img_rescaled = (2**16)*((img-min_pixel)/(max_pixel-min_pixel))
            imgs_rescaled.append(img_rescaled.astype('uint16'))

        for i in range(len(imgs_rescaled)):
            cv2.imwrite(output+str(int(energy))+str(120)+".tif",imgs_rescaled[120])
        

    plt.close()
    fig, axes = plt.subplots(1, 1)
    axes.set_title("Thickness at energy " + str(int(energy)) + " eV")
    axes.imshow(imgs_rescaled[120],cmap='gray')
    axes.set_xlabel("alpha = " + str(round(alpha,6)))
    plt.tight_layout()
    plt.show()

normalize()