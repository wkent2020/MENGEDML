import numpy as np 
import matplotlib.pyplot as plt
import cv2 

def watershed(image, kernelSize=5):
    '''
    Applies watershed method to segment image into droplets
    '''

    #Convert into RGB
    img = np.copy(image)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    #Apply canny edge detection to whole image
    edges = cv2.Canny(image,100,200)
    
    #Generate a rough thresholding
    _,proc = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

    kernel = np.ones((kernelSize,kernelSize),np.uint8)
    #Morphological closing of the rough threshold to weed out background
    closingThresh = cv2.morphologyEx(proc,cv2.MORPH_CLOSE,kernel)

    #Morphological closing on edges, 
    # since they are often different than threshold
    closingEdge = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)


    #Combine the two closings
    closing = np.logical_or(closingEdge,closingThresh).astype('uint8')*255

    #Define sure background and sure foreground and 
    # take their as unknown region to be categorized
    sure_bg = np.uint8(closing)
    sure_fg = np.uint8(edges)
    unknown = cv2.subtract(sure_bg,sure_fg)

    '''
    #print debugging
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Foreground")
    axes[0].imshow(sure_fg,cmap='gray')
    axes[1].set_title("Background")
    axes[1].imshow(sure_bg,cmap='gray')
    plt.show()
    #Print debugging
    plt.close()
    plt.imshow(unknown,cmap='gray')
    plt.show()
    '''

    #Define markers for the watershed method
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)

    img[markers == -1] = [255, 255, 255]

    blobs = []
    canvas = np.zeros(image.shape).astype('uint16')
    for marker in range(markers.max()):
        binary = np.equal(markers, canvas).astype('uint8')*255
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blobs.append(np.array(contours)[0])
        canvas += 1
    blobs = np.array(blobs)

    return img, markers, blobs

img = cv2.imread("individual_developers/jackson/120.tif",-1)

image, markers, blobs = watershed(img,9)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

fig, axes = plt.subplots(1, 2)
axes[0].set_title("Image")
axes[0].imshow(img,cmap='gray')
axes[1].set_title("Markers")
axes[1].imshow(markers,cmap='magma')
plt.show()

plt.close()
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
shapes_image = cv2.drawContours(img, blobs, -1, (255,0,0), 1)
plt.imshow(shapes_image) 
plt.show()