#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:26:43 2019

@author: williamkent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

# Generates and plots random data

X= -2 * np.random.rand(100,2)

X1 = 1 + 2 * np.random.rand(50,2)

X[50:100, :] = X1



# Fits data

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
Kmean.cluster_centers_

plt.scatter(X[ : , 0], X[ :, 1], s = 10, c = 'b')
plt.scatter(Kmean.cluster_centers_[0][0],Kmean.cluster_centers_[0][1],s=50, c='r')
plt.scatter(Kmean.cluster_centers_[1][0],Kmean.cluster_centers_[1][1],s=50, c='g')

plt.show()