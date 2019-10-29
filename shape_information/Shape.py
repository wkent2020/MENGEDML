import cv2
import numpy as np

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
    self.centroid (center of mass expressed relative to the upper right of the contour)

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
        self.centroid = [np.sum(inverseFloat * grids[dim].astype(float)) / mass 
                        for dim in range(inverseFloat.ndim)]
        #Might be worthwhile to normalize the center of mass coordinates
        
