import cv2
import numpy as np

num = "0001"

img = cv2.imread('Honda/80bar//T2-X=0.00_Y=1.00__'+num+".tif",-1)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)
cv2.imwrite("./norm_imgs/"+num+".png",img_rescaled)