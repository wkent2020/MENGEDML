#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:36:06 2019

@author: williamkent
"""

import cv2
import numpy as np
import json
import os

#image_folder = 'norm_imgs'
filepath = './Old_Groups_Code/Honda/80bar'

images = [img for img in os.listdir(filepath)]

print(images[0])

frame = cv2.imread(filepath + '/' + images[0])

print(frame.astype('uint8'))


cv2.imwrite(filepath + '/test.tif', frame.astype('uint8'))
