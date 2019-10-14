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

num = "0100"

img = cv2.imread('Old_Groups_Code/Honda/80bar/T2-X=0.00_Y=1.00__'+num+".tif",-1)
#img = cv2.imread('Old_Groups_Code/norm_imgs/100.tif',-1)

fig = plt.figure()
ax = fig.gca(projection='3d')

xlim = len(img[0])
ylim = len(img)

# Make data.
X = np.linspace(0, xlim-1, xlim)
Y = np.linspace(0, ylim-1, ylim)
X, Y = np.meshgrid(X, Y)


# Plot the surface.
#surf = ax.plot_surface(X, Y, img, cmap=cm.coolwarm,linewidth=0, antialiased=False)
surf = ax.scatter(X,Y,img)