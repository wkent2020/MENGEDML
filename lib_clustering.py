import numpy as np
import cv2
from lib_images import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
def maxdim(shapes):
    '''
    Global function to determine the maximum height and the maximum width of
      all the shapes in a directory.
    '''
    maxh = 0
    maxw = 0
    for shape in shapes:
        if shape.h > maxh:
            maxh = shape.h
        if shape.w > maxw:
            maxw = shape.w
    return(maxh,maxw)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
def kmeans(shapes,k=4):
    '''
    Run kmeans clustering on flattened shape images
    '''
    #prepare shapes
    maxh,maxw = maxdim(shapes)
    for shape in shapes:
        shape.pad(maxh,maxw)
        shape.flatten()

    rows = [shape.flat for shape in shapes]

    Z = np.stack(rows)
    Z = np.float32(Z)

    #uncomment these lines to run elbow method to determine the
    #best number of clusters for kmeans
    '''
    Elbow Method:
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(Z)
        Sum_of_squared_distances.append(km.inertia_)

    plt.figure()
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    '''

    #run kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    #assign a label to each shape
    for i in range(len(shapes)):
        shapes[i].setLabel(label[i])

    return(ret,label,center)
# ----------------------------------------------------------------------------
