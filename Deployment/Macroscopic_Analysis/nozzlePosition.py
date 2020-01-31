import cv2
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

from sprayEdges import *

def getNozzlePosition(binaryImages, rowLOC_list, columnLOC_list):
	Xs = []
	Ys = []
	for i in range(len(binaryImages)):
		leftRegression, rightRegression, linex, leftline, rightline, intersectionX, intersectionY = contours_to_points(binaryImages[i], rowLOC_list[i], columnLOC_list[i], 6, method='hist', data='black', plot=False, save=False)
		Xs.append(intersectionX)
		Ys.append(intersectionY)

	nozzleX = np.mean(Xs)
	nozzleY = np.mean(Ys)

	return nozzleX, nozzleY
	