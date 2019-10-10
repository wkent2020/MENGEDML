#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:36:06 2019

@author: williamkent
"""

import cv2
import os

image_folder = 'norm_imgs'
filepath = './Old_Groups_Code/H_Spray/150bar/T2'

images = [img for img in os.listdir(filepath)]

frame = cv2.imread(filepath + '/' + images[0])

cv2.imshow('image', frame)