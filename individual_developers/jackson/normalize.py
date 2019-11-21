import cv2
import numpy as np
from scipy import fftpack

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
    output = 'individual_developers/jackson/normalizedImages/8Bit/'
    background_frames = 6
    removeBackground = 1
    backgroundSection = [0,100,412,512]
    eightBit = 1
    fourier = 1
    alpha = 2*10**(3) / (3.1*10**(-5) * 250*10**(3))

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
        
        if fourier:
            img = fourierFilter(img, alpha)

        if removeBackground:
            img_nobg = img-bg
            #subtract background to bring background to 1
            second_bg = np.mean(np.mean(img_nobg[backgroundSection[0]:backgroundSection[1], \
                                backgroundSection[2]:backgroundSection[3]]))
            img_norm = img_nobg-second_bg
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
    if eightBit:


        for img in imgs_norm:
            img_rescaled = 255*((img-min_pixel)/(max_pixel-min_pixel))
            imgs_rescaled.append(img_rescaled.astype('uint8'))

        #write images -- this will automatically convert all values to uint8
        for i in range(len(imgs_rescaled)):
            cv2.imwrite(output+str(i)+".tif",imgs_rescaled[i])
    else:
        #Convert back to 16-bit after the background removal
        #Does nothing if the background is not removed
        for img in imgs_norm:
            img_rescaled = (2**16)*((img-min_pixel)/(max_pixel-min_pixel))
            imgs_rescaled.append(img_rescaled.astype('uint16'))

        for i in range(len(imgs_rescaled)):
            cv2.imwrite(output+str(i)+".tif",imgs_rescaled[i])

normalize()