#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:30:53 2019

@author: williamkent
"""
import cv2
import numpy as np

class Spray(object):
    '''
    Spray class: 
    
    Arguments: 
	
    Relevant Features:
    

    '''
    def __init__(self, locations, image):
        self.contour = contour
        self.image = image
        self.getApprox()
        self.getBoundary()
        self.getPerimeter()
        self.crop(image)
        self.shapeIntensity()