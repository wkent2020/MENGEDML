import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import colorsys
import os
import sys

# ----------------------------------------------------------------------------
## Global Functions
# ----------------------------------------------------------------------------
with open('input.json') as f:
  data = json.load(f)

def rotate_image(img):
  '''
  Rotate all images to align the spray vertically downards, and pad the images
    with white space.  This simplifies macroscopic image analysis
  '''
  with open('input.json') as f:
    data = json.load(f)
  ang = data["angle"]
  h,w = img.shape[:2]
  image_center = (w/2,h/2)

  rotation_mat = cv2.getRotationMatrix2D(image_center,ang,1.)

  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(h*abs_sin + w*abs_cos)
  bound_h = int(h*abs_cos + w*abs_sin)
  rotation_mat[0,2] += bound_w/2 - image_center[0]
  rotation_mat[1,2] += bound_h/2 - image_center[1]

  return cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))

def write(img,location):
    cv2.imwrite(location,img)

def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return (int(255*r), int(255*g), int(255*b))
 
def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## Image Class
# ----------------------------------------------------------------------------
class Image(object):
    '''
    Image class: takes an image file, converts it to a 2-D 8-bit matrix,
      allowing for a variety of operations and analyses
    '''

    def __init__(self, location): #location is str to image's directory
        self.location = location
        if data["normalize"]:
          self.front_index = len(data["normalized_img_dir"])
        else:
          self.front_index = int(location.find("/")) + int(data["label_common_length"])
        self.readImage()
        self.process()
        self.getShapes()
        self.drawShapes()

    def readImage(self):
        '''
        Reads an image file using the cv2 package into a greyscale 2-D 8-bit matrix
        '''
        #read image as 2d array (grayscale)
        if data["cut_bottom"]:
          self.image = cv2.imread(self.location, -1)[:-data["cut_bottom"]]
        if data["cut_top"]:
          self.image = cv2.imread(self.location, -1)[data["cut_top"]:]
        if data["cut_right"]:
          self.image = cv2.imread(self.location, -1)[:,:-data["cut_right"]]
        if data["cut_left"]:
          self.image = cv2.imread(self.location, -1)[:,data["cut_left"]:]
        self.image = rotate_image(self.image)
        print("Read image")
        abcdef = type(self.image[0][0])
        print(len(self.image))
        print(len(self.image[0]))
        print(abcdef)
        print(self.image[0][0])

    def process(self):
        '''
        Thresholds the image using single-value thresholding, Otsu thresholding, or
          multithresholding.  The image is binarized, turning all values above the 
          given threshold to white and all below to black
        '''
        with open('input.json') as f:
            data = json.load(f)
        if not(data["eightBit"]):
          #Send image to float32 so that openCV plays nicely
          self.image = (np.float32(1.0)*self.image/(2**16-1)).astype('float32')
        #smooth the image with a bilateral filter
        blur = cv2.bilateralFilter(self.image,9,150,150)
        #turn everything below (limit) to black
        if data["Otsu"] == 1:
            ret,proc = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif data["kthresh"] != 0:
            #get threshold value from kmeans clustering with k=kthresh
            Z = self.image.reshape((-1,1))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret,label,center=cv2.kmeans(Z,data["kthresh"],None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            # Now convert back into uint8, and make original image
            if data['eightBit']:
              center = np.uint8(center)
            self.center = center
            res = center[label.flatten()]
            self.kthreshed = res.reshape((self.image.shape))
            thresh_val = min(center)[0]
            print("kmeans multithreshold value of "+str(thresh_val))            
            #now threshold the k-means clustered image to only keep the darkest cluster
            ret,proc = cv2.threshold(self.kthreshed,thresh_val,255,cv2.THRESH_BINARY)
        else:
            ret,proc = cv2.threshold(blur,data["pixel_threshold"],255,cv2.THRESH_TOZERO)
            #turn everything above (limit) to white
            ret,proc = cv2.threshold(proc,data["pixel_threshold"],255,cv2.THRESH_TRUNC)
            if data['eightBit']:
              ret,proc = cv2.threshold(self.image,data["pixel_threshold"]-10,255,cv2.THRESH_BINARY)
            else:
              # this adjustment by 10 is pretty problematic
              ret,proc = cv2.threshold(self.image,data["pixel_threshold"]-10/255,255,cv2.THRESH_BINARY)
            
            #highlight the edges with Canny edge detection
            #proc = cv2.Canny(self.image,100,200)

        if not(data["eightBit"]):
          # threshold function outputs binary float image with values 0. and 255. 
          proc = np.uint8(proc)
          # return image to 16-bit
          self.image = (2**16*self.image).astype('uint16')
        self.binary = proc
        #highlight the edges with Canny edge detection
        self.processed_image = cv2.Canny(proc,100,200)
        self.processed_image = proc
        cv2.imwrite(data["binary_dir_name"] + "binary_" \
                        + str(self.location)[self.front_index:-4]+".jpg",proc)

    def getShapes(self):
        '''
        From binary images, find shapes which fit parameters assigned in json
        '''
        #find shapes in processed (binary) image
        contours, hierarchy = cv2.findContours(self.processed_image, cv2.RETR_TREE,\
cv2.CHAIN_APPROX_SIMPLE)
        self.shapes = [Shape(contour) for contour in contours]
        #apply parameters assigned in json
        big_shapes = []
        for shape in self.shapes:
            if shape.area>data["shape_area_pixel_minimum"] and \
              shape.area<data["shape_area_pixel_maximum"] and \
              (shape.h/float(shape.w)) < data["shape_side_ratio_maximum"] and \
              (shape.w/float(shape.h)) < data["shape_side_ratio_maximum"]:
                #checks if contour is closed (commented bc it removes vast
                #majority of shapes), uncomment if using Canny edge detection
                #if cv2.isContourConvex(shape.contour) == True: 
                big_shapes.append(shape)
        self.big_shapes = big_shapes

    def drawShapes(self):
        '''
        Draw contours onto images
        '''
        #draw big shapes over original image (or binary image)
        big_contours = [shape.contour for shape in self.big_shapes]        
        shapes_image = np.copy(self.binary)
        #change back to RGB for easier visualization
        shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
        self.shapes_image = cv2.drawContours(shapes_image, big_contours, -1, (0,0,255), 1 )
        #plt.imshow(self.shapes_image) # uncomment these lines to plot in real time
        #plt.show()

    def splitShapes(self, shapedir, eightBit):
        '''
        Save all shapes as individual images into a single directory 'shapedir'
        Saves shapes differently based on Boolean 'eightBit'
        '''
        #crop each shape
        shapes_split = [bshape.crop(self.image) for bshape in self.big_shapes]
        idx = 0
        for shape in shapes_split:
            idx += 1
            name = self.location[self.front_index:-4] + "_" + \
                    str(len(self.big_shapes))+ "_" + str(idx)
            #print(name)
            if eightBit:
              write(shape,shapedir+name+".jpg")
            else:
              cv2.imwrite(shapedir+name+".tif", shape)

    def writeLabels(self,target):
        '''
        Save kmeans cluster labels with each shape
        '''
        idx = 0
        for shape in self.big_shapes:
            idx+=1
            name = self.location[self.front_index:-4] + "_" + \
                    str(len(self.big_shapes))+ "_" + str(idx)
            loc = target+str(shape.label[0])+"/"+name+".jpg"
            write(shape.cropped, loc)

    def drawLabels(self):
        '''
        Draw shape contours on top of images, color coding by kmeans cluster
        '''
        clusters_image = np.copy(self.image)
        #change back to RGB for easier visualization
        clusters_image = cv2.cvtColor(clusters_image, cv2.COLOR_GRAY2RGB)
        self.clusters_image = None
        if self.big_shapes:
          k = np.amax([shape.label[0] for shape in self.big_shapes])+1 
          cluster_contours = [[] for i in range(k)]
          colors = getDistinctColors(k)
          for shape in self.big_shapes:
              cluster_contours[shape.label[0]].append(shape.contour)
          for i in range(len(cluster_contours)):
              clusters_image = cv2.drawContours(clusters_image,cluster_contours[i],-1,\
                              next(colors),1)
          self.clusters_image = clusters_image
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
## Shape Class
# ----------------------------------------------------------------------------
class Shape(object):
    '''
    Shape class: creates a shape object for purposes of saving shape
      contours with a variety of parameters for clustering and analysis
    '''
    def __init__(self, contour):
        self.contour = contour
        self.label = False
        self.getArea()
        self.getApprox()
        self.getBoundary()

    def getArea(self):
        '''
        Calculate area of shape
        '''
        self.area = cv2.contourArea(self.contour)

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

    def crop(self,parent):
        '''
        Crop shape from rest of image
        '''
        peri = cv2.arcLength(self.contour, True)
        #approximate shape of contour
        approx = cv2.approxPolyDP(self.contour, 0.02 * peri, True)
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
        return(self.cropped)
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

    def setLabel(self,label):
        '''
        Assign the kmeans cluster label to the shape
        '''
        self.label = label
# ----------------------------------------------------------------------------
