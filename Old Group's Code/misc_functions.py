import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from lib_images import *
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from lib_clustering import maxdim

def overlay(all_shapes):
    '''
    Takes as input all_shape variables from run.py
    Outputs overlayed plots (representative shapes of kmeans clusters)
    '''
    #Separating shapes with different labels
    shapes0 = [shapes for shapes in all_shapes if shapes.label == 0]
    shapes1 = [shapes for shapes in all_shapes if shapes.label == 1]
    shapes2 = [shapes for shapes in all_shapes if shapes.label == 2]
    add_l = []
    for i, allshapes in enumerate([shapes0, shapes1, shapes2]):
        maxh, maxw = maxdim(allshapes)
        #Initialize base layer
        add = np.uint8(np.full([maxh,maxw], 255))
        #alpha is opacity
        alpha = .02
        for shape in allshapes:
            h, w = shape.cropped.shape
            #padding to make all images the same dimension
            padh = (maxh - h) / 2
            padw = (maxw - w) / 2
            padshape = np.pad(shape.cropped, ((math.floor(padh), math.ceil(padh)), \
                (math.floor(padw), math.ceil(padw))), 'constant', constant_values = 255)
            #Overlay shape on base image
            add = cv2.addWeighted(padshape, alpha, add, 1-alpha, 0)
        add_l.append(add)
        plt.imshow(cv2.bitwise_not(add))
        plt.title('Shape overlay for cluster {}'.format(i))
        plt.show()
