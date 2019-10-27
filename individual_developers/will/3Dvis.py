#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:17:53 2019

@author: williamkent
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from contourTesting import drawShapes






num = "160"

img = cv2.imread('Old_Groups_Code/norm_imgs/'+num+".tif",-1)
#img = cv2.imread('Old_Groups_Code/norm_imgs/100.tif',-1)



xlim = len(img[0])
ylim = len(img)

# Make data.
X = np.linspace(0, xlim-1, xlim)
Y = np.linspace(0, ylim-1, ylim)
X, Y = np.meshgrid(X, Y)

finalmesh = []

for i in range(len(X)):
    for j in range(len(X[0])):
        finalmesh.append((X[i][j],Y[i][j],img[i][j]))
        

finalmesh2 = img.reshape(-1)
#print(finalmesh2)

finalmesh = np.float32(finalmesh)

finalmesh2 = np.float32(finalmesh2)

        
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(finalmesh,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
ret2,label2,center2=cv2.kmeans(finalmesh2,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
center2 = np.uint8(center2)

res = center[label.flatten()]
res = np.array([r[2] for r in res])
#Takes 1-D array and puts back into image format
kthreshed = res.reshape((img.shape))
#Lowest grayscale k-means img: aka some color
thresh_val = min(center)[0]
print("kmeans multithreshold value of "+str(thresh_val))            
#now threshold the k-means clustered image to only keep the darkest cluster
ret,proc = cv2.threshold(kthreshed,thresh_val,255,cv2.THRESH_BINARY) #binarizes
#Binary image file
print(proc)
#Threshold value
print(ret)


res2 = center2[label.flatten()]
kthreshed2 = res.reshape((img.shape))
thresh_val2 = min(center2)[0]
print("kmeans multithreshold value of "+str(thresh_val2))            
#now threshold the k-means clustered image to only keep the darkest cluster
ret2,proc2 = cv2.threshold(kthreshed2,thresh_val2,255,cv2.THRESH_BINARY) #binarizes
#Binary image file
print(proc2)
#Threshold value
print(ret2)

image_binarized = proc
image = img


contours, hierarchy = cv2.findContours(image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
shapes_image = np.copy(image)
#change back to RGB for easier visualization
shapes_image = cv2.cvtColor(shapes_image, cv2.COLOR_GRAY2RGB)
#plt.imshow(shapes_image) # uncomment these lines to plot in real time
#plt.show()
shapes_image = cv2.drawContours(shapes_image, contours, -1, (255,0,0), 1)

image_binarized2 = proc
image2 = img

contours2, hierarchy2 = cv2.findContours(image_binarized2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
shapes_image2 = np.copy(image2)
#change back to RGB for easier visualization
shapes_image2 = cv2.cvtColor(shapes_image2, cv2.COLOR_GRAY2RGB)
#plt.imshow(shapes_image) # uncomment these lines to plot in real time
#plt.show()
shapes_image2 = cv2.drawContours(shapes_image2, contours2, -1, (255,0,0), 1)


plt.subplot(1,2,1)
plt.imshow(shapes_image) 
plt.title("3D K-means")
plt.subplot(1,2,2)
plt.imshow(shapes_image2) 
plt.title("Regular")
plt.show()




#plt.close('all')
#fig = plt.figure()
#ax = fig.gca(projection='3d')

# Plot the surface.
#surf = ax.plot_surface(X, Y, img, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#surf = ax.scatter(X,Y,img)