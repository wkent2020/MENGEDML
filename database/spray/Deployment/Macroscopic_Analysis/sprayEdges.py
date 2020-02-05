import cv2
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def get_hist_edge(freq, x, elist, lim, left=True):
	if left:
		i = int(len(freq)*0.4)
		while freq[i] > lim:
			i -= 1
			if i < 0:
				break
		if i >= 0:
			elist.append(x[i])
	else:
		i = int(len(freq)*0.66)
		while freq[i] > lim:
			i += 1
			if i == len(freq)-1:
				break
		if i != len(freq)-1:	
			elist.append(x[i+1])

def linetoangle(left,right):
	'''
	linetoangle function: takes linear regression information 
		and returns tuple of spray angle, left side of spray angle,
		and right side of spray angle. 
	'''
	left = np.arctan(1/left[0])
	right = np.arctan(1/right[0])
	opening = left+right
	return (opening,left,right)
	
def contours_to_points(img, rowLOC, columnLOC, regpoints, method='simple', data='centers', plot=False, save=False):
	'''
	
	'''
	top = len(img)
	width = len(img[0])
	newlocs = []
	if data == 'centers':
		conty = rowLOC
		contx = columnLOC
		# Combines corrected y-values with x-values
		for i in range(len(conty)):
			newlocs.append([contx[i],conty[i]])
			
	elif data == 'black':
		for i in range(top):
			for j in range(width):
				if img[i][j] == 0:
					newlocs.append([j,i])
		
		conty = [c[1] for c in newlocs]
		contx = [c[0] for c in newlocs]

	intervals = [0]
	intermid = []
	div_val = int(top/regpoints)
	
	# Computes vertical divisions
	for i in range(regpoints):
		intervals.append(div_val*(i+1))
		intermid.append(div_val*(i+1)-(div_val*(i+1)-intervals[i])/2)
	
	# Creates nested list for points
	separated_contours = []
	for i in range(regpoints):
		separated_contours.append([])
	
	# Separates points into vertical intervals
	for i in range(regpoints):
		for j in range(len(newlocs)):
			if (newlocs[j][1] >= intervals[i]) and (newlocs[j][1] <= intervals[i+1]):
				separated_contours[i].append(newlocs[j][0])
		
	left = []
	right = []
	if method == 'simple':
		for i in range(regpoints):
			mean = np.mean(separated_contours[i])
			std = np.std(separated_contours[i])
			left.append(mean - 1.5*std)
			right.append(mean + 1.5*std)
	
	if method == 'hist':
		lcount = 0
		rcount = 0
		if data == 'black':
			limstart = 20
		else:
			limstart = 2
		for i in range(regpoints):
			freq, x = np.histogram(separated_contours[i], bins=50)
			lim = limstart
			
			while len(left) == lcount:
				get_hist_edge(freq, x, left, lim)
				lim += 1
			lim = limstart
			lcount += 1
			while len(right) == rcount:
				get_hist_edge(freq, x, right,lim,left=False)
				lim += 1
			rcount += 1
	
	leftedge = linregress(left,intermid)
	rightedge = linregress(right,intermid)
	
	sangle, langle,rangle = linetoangle(leftedge,rightedge)

	linex = np.linspace(0,width,1000)
	leftline = [(leftedge[0]*x)+leftedge[1] for x in linex]
	rightline = [(rightedge[0]*x)+rightedge[1] for x in linex]
	dif  = np.abs(np.array(leftline)-np.array(rightline))
	difIndex = list(dif).index(np.min(np.abs(np.array(leftline) - np.array(rightline))))
	intersectionY = leftline[difIndex]
	intersectionX = linex[difIndex]

	if plot:
		plt.imshow(img,cmap='gray')
		plt.scatter(right,intermid, color='red')
		plt.scatter(left,intermid, color='red')
		plt.plot(linex,leftline,'r--')
		plt.plot(linex,rightline,'r--')
		plt.ylim(top,0)
		plt.xlim(0,width)
		plt.title("Contoured Imaged")

	return leftedge, rightedge, linex, leftline, rightline, intersectionX, intersectionY
