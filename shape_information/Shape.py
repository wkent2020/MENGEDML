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
    
    def pullSeeds(self, seeds = None):
        '''
        Take a binary image of seeds and define markers as input for the watershed method
        Argument 'seeds' should be the binary image over the entire image taken by thresholding
        '''
        
        #After seeds are passed, this method will reconstruct the original markers
        if seeds is not None:
            #Crop the image
            self.peaks = seeds[self.boundary[0]:self.boundary[1],\
                        self.boundary[2]:self.boundary[3]]
        croppedPeaks = cv2.bitwise_or(self.peaks,self.cropped)
        #Invert the binary image
        _, self.seeds = cv2.threshold(croppedPeaks,254,255,cv2.THRESH_BINARY_INV) 

        _, self.markers = cv2.connectedComponents(self.seeds)
        #Find connected components
        self.markers += 1
        
    def pullSeeds(self):
        '''
        Overloaded pullSeeds to recover original markers if necessary
        '''
        croppedPeaks = cv2.bitwise_or(self.peaks,self.cropped)
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

    def childPad(self):
        '''
        Pad the child shape to allow for easier comparisons
        '''
        maxSideY = self.image.shape[0] - self.boundary[1] 
        maxSideX = self.image.shape[1] - self.boundary[3]
        self.paddedChild = np.pad(self.cropped,((self.boundary[0], maxSideY), \
                      (self.boundary[2], maxSideX)), 'constant', constant_values=255)

    def divideShapes(self):
        '''
        Divide the shape into children after application of watershed
        '''
        #Split shape via markers
        self.split = np.copy(self.cropped)
        self.split[self.markers == -1] = 255

        self.children = []
        canvas = np.zeros(self.cropped.shape).astype('uint16')+2
        # For each marker, contour into a new shape
        for marker in range(self.markers.max()-1):
            binary = np.equal(self.markers, canvas).astype('uint8')*255
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #Save the contours in a list of children
            for contour in contours:
                child = Shape(contour, self.cropped)
                child.childPad()
                self.children.append(child)
            canvas += 1

    def childCriteria(self, child):
        '''
        Set some criteria for a bad shape
        '''
        minSize = 5
        if minSize > child.area:
            return True
        else:
            return False

    def splitRefine(self, peaks = None, recursive = False):
        '''
        Recursively refine the split using the criteria function to remove small droplets
        
        Can modify this function and criteria to screen the watershed differently
        As written, overwrites original watershed info, but 
        that information can be recovered using the overloaded pullSeeds
        '''
        if peaks is None:
            peaks = np.copy(self.peaks)
        modify = 0
        for child in self.children:
            if self.childCriteria(child):
                croppedPeaks = cv2.bitwise_or(peaks,child.paddedChild)
                _, seeds = cv2.threshold(croppedPeaks,254,255,cv2.THRESH_BINARY_INV)
                peaks = cv2.bitwise_or(peaks, seeds) 
                modify = 1
        if modify:
            croppedPeaks = cv2.bitwise_or(peaks,self.cropped)
            #Invert the binary image
            _, self.seeds = cv2.threshold(croppedPeaks,254,255,cv2.THRESH_BINARY_INV) 

            _, self.markers = cv2.connectedComponents(self.seeds)
            #Find connected components
            self.markers += 1
            self.watershed()
            self.divideShapes()
            if recursive:
                self.splitRefine(peaks=peaks)


    def dropDivide(self, seeds):
        '''
        Single function call to divide drops using watershed based on argument seeds
        '''        
        self.pullSeeds(seeds)
        #Check for multiple connected components
        if self.markers.max() > 2:
            self.watershed()
            self.divideShapes()
            #self.splitRefine()
