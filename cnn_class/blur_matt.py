import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('lena.png')
plt.imshow(img)
plt.show()

bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50)

plt.imshow(W, cmap='gray')
plt.show()

out = convolve2d(bw, W)
plt.imshow(out, cmap='gray')
plt.show()

# convoluted output is different shape
print(out.shape)
print(bw.shape)

# keep shape same after convolution
out = convolve2d(bw, W, mode='same')
plt.imshow(out, cmap='gray')
plt.show()

out3 = np.zeros(img.shape)
for i in range(3):
    out3[:, :, i] = convolve2d(img[:, :, i], W, mode='same')
out3 /= out3.max()
plt.imshow(out3)
plt.show()
