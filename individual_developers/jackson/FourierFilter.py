import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

img = cv2.imread("individual_developers/jackson/120.tif",-1)
#Material parameters to be determined later
mu = 2*10**(3)
zDelta = 3.1*10**(-5) * 250*10**(3)
alpha = mu/zDelta

'''
img = cv2.imread("individual_developers/jackson/jmi_1010_f3b.tiff",-1)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('uint32')
#Material parameters to be determined later
mu = 0.11322 *10**(-2)
zDelta = -0.375 *250
alpha = zDelta/mu
'''

#Fourier frequency coordinates
ky = fftpack.fftfreq(len(img))
kx = fftpack.fftfreq(len(img[0]))
kx, ky = np.meshgrid(kx, ky)
#Compute their magnitudes
fourierMagnitude = np.square(kx) + np.square(ky)

img_ft = fftpack.fft2(img)

transformedFT = img_ft/ (1+ alpha*fourierMagnitude)

transformedImage = fftpack.ifft2(transformedFT).real

thickness = -(1/mu)*np.log(transformedImage)

plt.close()
fig, axes = plt.subplots(1, 2)
axes[0].set_title("Original")
axes[0].imshow(img,cmap='gray')
axes[1].set_title("Thickness")
axes[1].imshow(thickness,cmap='gray')
axes[1].set_xlabel("alpha =" + str(alpha))
plt.tight_layout()
plt.show()

