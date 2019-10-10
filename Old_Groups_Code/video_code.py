#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:43:32 2019

@author: williamkent
"""

import cv2
import os

image_folder = 'norm_imgs'
video_name = 'video2.mp4'

images_num = [i for i in range(0,399)]
images = []

for i in range(0,399):
    images.append(str(images_num[i]) + '.tif')

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

ourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, ourcc, 12, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()