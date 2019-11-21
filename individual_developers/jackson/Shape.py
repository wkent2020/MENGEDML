import cv2
import numpy as np
import matplotlib.pyplot as plt

class Shape(object):
    '''
    Shape class: creates a shape object for purposes of saving shape
      contours with a variety of parameters for clustering and analysis
    
    Arguments: contour object from cv2.findContours, parent image as 2D array
	
    Relevant Features:
    self.area (calculated from number of pixels in contoured image)
    self.perimeter (calculated from cv2.arcLength)
    self.boundary (boundary of bounding rectangle around contour)
    self.centerLocation (center of bounding rectangle)
    self.cropped (cropped contour image from parent image)
    self.bordered (cropped image padded with 2 pixels of white space on borders)
    self.meanDensity (average greyscale value of pixels in contour)
    self.shapePixels (list of greyscale pixel values of the contour shape (white backgound removed))
    self.centroid (center of mass expressed in image coordinates)

    '''
    def __init__(self, contour, image):
        self.contour = contour
        self.image = image
        self.getApprox()
        self.getBoundary()
        self.getPerimeter()
        self.crop(image)
        self.shapeIntensity()

    def getPerimeter(self):
        perimeter = cv2.arcLength(self.contour, True)
        self.perimeter = perimeter

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
        self.centerLocation = [(y + h/2.0), (x + w/2.0)]

    def crop(self, parent):
        '''
        Crop shape from rest of image
        '''
        #approximate shape of contour
        approx = cv2.approxPolyDP(self.contour, 0.02 * self.perimeter, True)
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

    def shapeIntensity(self):
    	self.shapePixels  = list(filter(lambda x: x != 255, self.cropped.flatten()))
    	self.area = len(self.shapePixels)
    	self.meanDensity = np.sum(self.shapePixels) / self.area

    def centerOfMass(self):
        '''
        Compute the center of mass relative to the contour coordinates
        '''
        #Convert the intensity to a float with 0 as light and 1 as dark
        inverseFloat = (self.cropped - 255)/-255 
        #Compute total mass
        mass = np.sum(inverseFloat) 
        grids = np.ogrid[[slice(0,i) for i in inverseFloat.shape]]
        #Take weighted average of the mass
        #Outputes center of mass relative to the coordinates of the contour
        centroid = [np.sum(inverseFloat * grids[dim].astype(float)) / mass 
                        for dim in range(inverseFloat.ndim)]
        #Convert to image coordinates
        self.centroid = [centroid[0]+self.boundary[0], centroid[1]+self.boundary[2]]
    
    def pullSeeds(self, seeds):
        '''
        Take a binary image of seeds and define markers as input for the watershed method
        Argument 'seeds' should be the binary image over the entire image taken by thresholding
        '''
        
        #Crop the image
        peaks = seeds[self.boundary[0]:self.boundary[1],\
                      self.boundary[2]:self.boundary[3]]
        croppedPeaks = cv2.bitwise_or(peaks,self.cropped)
        if croppedPeaks.min() == 255:
            self.invert = True
        else:
            self.invert = False
        if self.area > 50:
            fig, axes = plt.subplots(1, 2)
            axes[0].set_title("Cropped")
            axes[0].imshow(self.cropped,cmap='gray')
            axes[1].set_title("Peaks")
            axes[1].imshow(croppedPeaks,cmap='gray')
            plt.tight_layout()
            plt.show()
            plt.close()
        #Invert the binary image
        _, self.seeds = cv2.threshold(croppedPeaks,254,255,cv2.THRESH_BINARY_INV) 

        _, self.markers = cv2.connectedComponents(self.seeds)
        #Find connected components
        self.markers += 1
        
    def watershed(self):
        '''
        Divide the shapes via watershed after processing with pullSeeds
        '''
        #Invert the cropped contour into binary to form the background
        _,background = cv2.threshold(self.cropped,254,255,cv2.THRESH_BINARY_INV)
        #Form the unknown portion to be split into droplets
        unknown = cv2.subtract(background,self.seeds)
        self.markers[unknown==255] = 0
        #Change to color
        color = cv2.cvtColor(self.cropped,cv2.COLOR_GRAY2RGB)
        #Watershed
        self.markers = cv2.watershed(color,self.markers)

    def divideShapes(self):
        '''
        Divide the shape into children after application of watershed
        '''
        #Split shape via markers
        self.split = np.copy(self.cropped)
        self.split[self.markers == -1] = 255

        fig, axes = plt.subplots(1, 2)
        axes[0].set_title("Split")
        axes[0].imshow(self.split,cmap='gray')
        axes[1].set_title("Markers")
        axes[1].imshow(self.markers,cmap='gray')
        plt.tight_layout()
        plt.show()
        plt.close()

        self.children = []
        canvas = np.zeros(self.cropped.shape).astype('uint16')+2
        # For each marker, contour into a new shape
        for marker in range(self.markers.max()-1):
            binary = np.equal(self.markers, canvas).astype('uint8')*255
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                self.children.append(Shape(contour, self.cropped))
            canvas += 1

    def dropDivide(self, seeds):
        '''
        Single function call to divide drops using watershed based on argument seeds
        '''        
        self.pullSeeds(seeds)
        if self.markers.max() > 2:
            self.watershed()
            self.divideShapes()
            fig, axes = plt.subplots(1, 2)
            axes[0].set_title("Contour")
            axes[0].imshow(self.cropped,cmap='gray')
            axes[1].set_title("Markers")
            axes[1].imshow(self.markers,cmap='gray')
            plt.tight_layout()
            plt.show()
            plt.close()
