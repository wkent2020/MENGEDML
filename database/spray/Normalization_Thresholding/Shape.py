import cv2
from scipy.ndimage.measurements import center_of_mass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def imagePad(image, top, bottom, left, right, greyscale = 255):
    rows, columns = image.shape
    padTopBottom = np.zeros(columns) + greyscale
    padRight = np.zeros(rows) + greyscale
    for i in range(top):
        image = np.vstack((padTopBottom, image))
    for i in range(bottom):
        image = np.vstack((image, padTopBottom))
    padRight = np.array([np.zeros(rows+top+bottom) + greyscale])
    for i in range(left):
        image = np.insert(image, 0, greyscale, axis=1)
    for i in range(right):
        image = np.concatenate((image, padRight.T), axis=1)
    return image

class Shape(object):
    '''
    Shape class: creates a shape object for purposes of saving shape
      contours with a variety of parameters for clustering and analysis
    
    Arguments: contour object from cv2.findContours, parent image as 2D array
    
    '''

    def __init__(self, contour, image, binaryImage):
        self.contour = contour
        self.image = image
        self.binaryImage = binaryImage
        self.getApprox()
        self.getBoundary()
        self.getPerimeter()
        self.crop(self.image.copy(), 1)
        self.binaryCrop(binaryImage)
        self.getWhiteFringes()
        self.shapeIntensity()
        self.pixelLocations()
        self.moment_of_inertia()
        self.getWhiteFringes()
        self.fitEllipse()

    def imagePad(self, image, top, bottom, left, right, greyscale = 255):
        rows, columns = image.shape
        padTopBottom = np.zeros(columns) + greyscale
        padRight = np.zeros(rows) + greyscale
        for i in range(top):
            image = np.vstack((padTopBottom, image))
        for i in range(bottom):
            image = np.vstack((image, padTopBottom))
        padRight = np.array([np.zeros(rows+top+bottom) + greyscale])
        for i in range(left):
            image = np.insert(image, 0, greyscale, axis=1)
        for i in range(right):
            image = np.concatenate((image, padRight.T), axis=1)
        return image


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
        self.centerLocation = [(y + h/2.0), (x + w/2.0)] #row, col

    def crop(self, parentImage, border):
        '''
        Crop shape from rest of image
        '''
        self.border = border
        self.croppedImage = parentImage[(self.boundary[0]) : (self.boundary[1]), (self.boundary[2]) : (self.boundary[3])]

        #Checking if bounding rectangle is at edge of image frame--this wouln't allow us to pad the image correctly
        edgeOfFrame = np.array(self.boundary).copy()
        for i in range(len(edgeOfFrame)):
            if i == 1:
                if edgeOfFrame[i] == self.image.shape[0]:
                    edgeOfFrame[i] = 0
            if i == 3:
                if edgeOfFrame[i] == self.image.shape[1]:
                    edgeOfFrame[i] = 0
        edgeOfFrame[edgeOfFrame != 0] = 1

        self.edgeOfFrame = edgeOfFrame

        borders = self.edgeOfFrame * self.border
        self.borderedImage = parentImage[(self.boundary[0] - borders[0]) : (self.boundary[1] + borders[1]), (self.boundary[2] -borders[2]) : (self.boundary[3] + borders[3])]


    def binaryCrop(self, parent):
        '''
        Crop Binary Shape from rest of image
        '''
        #approximate shape of contour
        approx = cv2.approxPolyDP(self.contour, 0.02 * self.perimeter, True)
        # create a single channel pixel white image
        canvas = np.zeros(parent.shape).astype(parent.dtype) + 255
        fill = cv2.fillPoly(canvas, pts =[self.contour], color=0)
        #keep shape in grayscale, turn background white
        anti_fill = cv2.bitwise_or(parent,fill)
        self.binaryCropped = anti_fill[self.boundary[0]:self.boundary[1],\
                      self.boundary[2]:self.boundary[3]]

        #also crop to slightly larger than boundary so shape isn't right at 
        #the edge of the image
        #this will be useful if we want to draw more contours on a shape after cropping it
        border = 2
        bordered = anti_fill[self.boundary[0]-\
                        border:self.boundary[1]+\
                        border,self.boundary[2]-\
                        border:self.boundary[3]+border]
    
    def getWhiteFringes(self):
        kernel = np.ones((5,5),np.uint8)

        expandedCrop = (self.imagePad(self.binaryCropped.copy(), self.edgeOfFrame[0],self.edgeOfFrame[1],self.edgeOfFrame[2],self.edgeOfFrame[3]))
        self.dilation = cv2.erode(expandedCrop.copy(), kernel,iterations = 1)

        expandedCrop[expandedCrop == 0] = 1
        expandedCrop[expandedCrop == 255]  = 0

        bitwise_dilation = self.dilation.copy()
        bitwise_dilation[bitwise_dilation == 0] = 1
        bitwise_dilation[bitwise_dilation == 255] = 0

        whiteBorder = bitwise_dilation - expandedCrop

        self.whiteFringes = whiteBorder * self.borderedImage #make sure to fix borderedImage function
        self.maxWhiteFringe = np.max(self.whiteFringes)


    def moment_of_inertia(self):
        bitwise_shapeImage = self.binaryCropped.copy()
        bitwise_shapeImage[bitwise_shapeImage==0] = 1
        bitwise_shapeImage[bitwise_shapeImage == 255] = 0

        relativeCenterLocation = center_of_mass(bitwise_shapeImage.copy())
        self.COMLocation = relativeCenterLocation[0] + self.boundary[0], relativeCenterLocation[1] + self.boundary[2] #row, col

        grid = np.indices((bitwise_shapeImage.shape[0], bitwise_shapeImage.shape[1]))
        
        row_grid, col_grid = grid[0], grid[1]
        deltaRow_squared = (row_grid - relativeCenterLocation[0])**2.0
        deltaCol_squared = (col_grid - relativeCenterLocation[1])**2.0
        sum_squared_distances = (deltaRow_squared + deltaCol_squared) * bitwise_shapeImage

        self.moment_inertia = np.sum(sum_squared_distances) #sum of squared distances from center
        self.radius_gyration = (self.moment_inertia / self.area)**0.5


    def pixelLocations(self):
        pixels = np.argwhere(self.binaryCropped == 0) #(x,y) gives row, column in cropped image (unreferenced to larger image)
        row, col = self.boundary[0], self.boundary[2] 
        reference = [row, col]
        self.pixelsLocation = pixels + reference

    def flatten(self):
        '''
        Flatten the 2-D array of the shape image into a 1-D array
        '''
        #when clustering, each shape is represented as a single row of values
        #only flatten AFTER padding
        self.flat = np.ndarray.flatten(self.padded)

    def shapeIntensity(self):
        self.shapePixels  = list(filter(lambda x: x != 255, self.binaryCropped.flatten()))
        self.area = len(self.shapePixels)

        bitwise_cropped = self.binaryCropped.copy()
        bitwise_cropped[bitwise_cropped == 0] = 1
        bitwise_cropped[bitwise_cropped == 255] = 0
        self.cumulativeMass = np.sum(bitwise_cropped * self.croppedImage)
        pixelImage = (bitwise_cropped * self.croppedImage).reshape(1, bitwise_cropped.shape[0] * bitwise_cropped.shape[1])
        pixelImage = pixelImage[pixelImage != 0]
        self.pixelSD = np.std(pixelImage)

    def fitEllipse(self):
        if len(self.contour) >= 5:

            self.ellipse = cv2.fitEllipse(self.contour)
            (x,y),(MA,ma),angle = self.ellipse
            self.majorAxis = MA
            self.minorAxis = ma
            self.ellipseAngle = angle
            self.ellipseAspectRatio = MA/ma
        else:
            self.majorAxis = -1
            self.minorAxis = -1
            self.ellipseAngle = -1
            self.ellipseAspectRatio = -1



def NestedShapes(dataFrame, image, binaryImage):
    contours = dataFrame['Contours']
    featureLabels = ['Pixel Locations','Shape Area', 'Shape Mass', 'Outer Perimeter', 'Inner Perimeter', 'Row LOC', 'Column LOC', 'ROW COM LOC', 'COL COM LOC', 'Moment of Inertia (Unweighted)', 'Radius of Gyration (Unweighted)', 'White Fringe Max', 'Pixel SD', 'Major Axis', 'Minor Axis', 'Ellipse Aspect Ratio', 'Ellipse Angle']

    sortedFrame = dataFrame.sort_values(['Generation'], axis=0, ascending=False, inplace=False)
    sortedContours = list(sortedFrame['Contours'])

    shapeFrame = pd.DataFrame(columns=featureLabels, index=sortedFrame.index)

    def getHoleShape(contour):
        hole = Shape(contour, image, binaryImage)
        hole_area = -1
        hole_mass = -1
        hole_outerperimeter = hole.perimeter
        hole_innerperimeter = -1
        hole_rowLOC, hole_colLOC = hole.centerLocation[0], hole.centerLocation[1]
        return [hole_area, hole_mass, hole_outerperimeter, hole_innerperimeter, hole_rowLOC, hole_colLOC, hole.COMLocation[0], hole.COMLocation[1],-1, -1, -1, -1, -1, -1, -1, -1]

    def getInnerShape(contour, shapeFrame, children, grandchildren):
        subShape = Shape(contour, image.copy(), binaryImage.copy())
        
        gc_area = 0
        gc_mass = 0
        for j in grandchildren:
            try:
                gc_area += shapeFrame['Shape Area'][j]
                gc_mass += shapeFrame['Shape Mass'][j]
            except:
                print('Hierarchy Error')
        subShape_area = subShape.area - gc_area
        subShape_mass = subShape.cumulativeMass - gc_mass

        c_peri = 0
        for j in children:
            try:
                c_peri += shapeFrame['Outer Perimeter'][j]
            except:
                print('Hierarchy Error')
        subShape_innerperimeter = c_peri
        
        subShape_outerperimeter = subShape.perimeter
        subShape_rowLOC, subShape_colLOC = subShape.centerLocation[0], subShape.centerLocation[1]
        subShape_pixelLocations = subShape.pixelsLocation

        shape_image = subShape.binaryCropped.copy()

        grandchildrenPixels = []
        for j in grandchildren:
            try:
                grandchildrenPixels = shapeFrame['Pixel Locations'][j]
                grandchildrenPixels -= [subShape.boundary[0], subShape.boundary[2]]
            except:
                print('Hierarchy Error')
            for i in range(len(grandchildrenPixels)):
                try:
                    shape_image[grandchildrenPixels[i][0], grandchildrenPixels[i][1]] = 255
                except:
                    print('Hierarchy Error--Check to verify solution.')
       
        #now, shape_image contains the binary image of just the parent shape, no children/grandchildren

        #get white fringes for just the shape_image, i.e. for just the parent shape and not the grandchildren
        kernel = np.ones((5,5),np.uint8)

        expandedCrop = (imagePad(shape_image.copy(), subShape.edgeOfFrame[0],subShape.edgeOfFrame[1],subShape.edgeOfFrame[2],subShape.edgeOfFrame[3]))
        dilation = cv2.erode(expandedCrop.copy(), kernel,iterations = 1)

        expandedCrop[expandedCrop == 0] = 1
        expandedCrop[expandedCrop == 255]  = 0
        bitwise_dilation = dilation.copy()
        bitwise_dilation[bitwise_dilation == 0] = 1
        bitwise_dilation[bitwise_dilation == 255] = 0
        whiteBorder = bitwise_dilation - expandedCrop

        whiteFringes = whiteBorder * subShape.borderedImage #make sure to fix borderedImage function
        maxWhiteFringe = np.max(whiteFringes)

        bitwise_shapeImage = shape_image.copy()
        bitwise_shapeImage[bitwise_shapeImage==0] = 1
        bitwise_shapeImage[bitwise_shapeImage == 255] = 0

        #Calculate variance of pixels contained in parent shape
        parentShape = bitwise_shapeImage * subShape.croppedImage #parentShape contains the pixels of just the parent shape, in their (nonbinary) float format
        parentPixels = parentShape.reshape(1, parentShape.shape[0] * parentShape.shape[1])
        parentPixels = parentPixels[parentPixels != 0]
        pixelSD= np.std(parentPixels)

        #Calculation of radius of gyration / moment of inertia (mass is unweighted--here, all mass = 1)
        relativeCenterLocation = center_of_mass(bitwise_shapeImage.copy())
        subShape_COMLOC = relativeCenterLocation[0] + subShape.boundary[0], relativeCenterLocation[1] + subShape.boundary[2]
        grid = np.indices((bitwise_shapeImage.shape[0], bitwise_shapeImage.shape[1]))
        row_grid, col_grid = grid[0], grid[1]
        deltaRow_squared = (row_grid - relativeCenterLocation[0])**2.0
        deltaCol_squared = (col_grid - relativeCenterLocation[1])**2.0
        sum_squared_distances = (deltaRow_squared + deltaCol_squared) * bitwise_shapeImage
        moment_inertia = np.sum(sum_squared_distances) #sum of squared distances from center
        radius_gyration = (moment_inertia / subShape_area)**0.5

        ellipseAspectRatio = subShape.ellipseAspectRatio
        majorAxis = subShape.majorAxis
        minorAxis = subShape.minorAxis
        ellipseAngle = subShape.ellipseAngle

        return [subShape_pixelLocations, subShape_area, subShape_mass, subShape_outerperimeter, subShape_innerperimeter, subShape_rowLOC, subShape_colLOC, subShape_COMLOC[0], subShape_COMLOC[1], moment_inertia, radius_gyration, maxWhiteFringe, pixelSD, majorAxis, minorAxis, ellipseAspectRatio, ellipseAngle]

    generations = list(sortedFrame['Generation'])
    for i in range(len(generations)):
        if generations[i]%2 == 1: #if generation is odd (hole)
            holeShape = getHoleShape(sortedContours[i])
            shapeFrame.iloc[i] = [None] + holeShape #pixel lcoations, and shape vector #fill in dataFrame with shape information at index i
        else:
            innerShape = getInnerShape(sortedContours[i], shapeFrame, sortedFrame['Children'].iloc[i], sortedFrame['Grandchildren'].iloc[i])
            shapeFrame.iloc[i] = innerShape

    shapeFrame.drop('Pixel Locations',axis= 1, inplace=True)
    shapeFrame.sort_index(axis = 0, inplace = True)
    return np.array(shapeFrame) #return just the shape information, stored as array of feature vectors


