import cv2
import numpy as np
import matplotlib.pyplot as plt
num = "0146"
n = "146"


img = cv2.imread('Honda/80bar//T2-X=0.00_Y=1.00__'+num+".tif",-1)
img_max = img.max()
img_min = img.min()
img_rescaled = 255*((img-img_min)/(img_max-img_min))
img_rescaled = np.array(img_rescaled, dtype = int)


img_rescaled = cv2.imread("./norm_imgs/" + n + ".tif", -1)


Z = img_rescaled.reshape((-1,1))
Z = img.reshape((-1,1))
Z = np.float32(Z)
Z = np.concatenate(Z)
bins = img_max - img_min

fig, axs = plt.subplots(1, 1)

axs.hist(Z, bins)
#axs.set_xlim(0,255)
plt.show()


cv2.imwrite("./norm_imgs/"+num+".png",img_rescaled)