import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

dssim = np.load('dssim.npy')
dsdsim = np.load('dsdsim.npy')

x01 = np.log(dssim[:,:,:,0] / dssim[:,:,:,1])
x02 = np.log(dssim[:,:,:,0] / dssim[:,:,:,2])
x12 = np.log(dssim[:,:,:,1] / dssim[:,:,:,2])

plt.subplot(3,3,1)
plt.hist2d(x01[~np.isnan(x01)], x02[~np.isnan(x02)], bins=16)

plt.subplot(3,3,2)
plt.hist2d(x01[~np.isnan(x01)], x12[~np.isnan(x12)], bins=16)

plt.subplot(3,3,3)
plt.hist2d(x02[~np.isnan(x01)], x12[~np.isnan(x12)], bins=16)

x01 = np.log(dsdsim[:,:,:,0] / dsdsim[:,:,:,1])
x02 = np.log(dsdsim[:,:,:,0] / dsdsim[:,:,:,2])
x12 = np.log(dsdsim[:,:,:,1] / dsdsim[:,:,:,2])

plt.subplot(3,3,4)
plt.hist2d(x01[~np.isnan(x01*x02)], x02[~np.isnan(x01*x02)], bins=16)

plt.subplot(3,3,5)
plt.hist2d(x01[~np.isnan(x01*x12)], x12[~np.isnan(x01*x12)], bins=16)

plt.subplot(3,3,6)
plt.hist2d(x02[~np.isnan(x01*x12)], x12[~np.isnan(x01*x12)], bins=16)

gamma = np.log(2) / 60 * 10
phi = dsdsim + gamma * dssim
x01 = np.log(phi[:,:,:,0] / phi[:,:,:,1])
x02 = np.log(phi[:,:,:,0] / phi[:,:,:,2])
x12 = np.log(phi[:,:,:,1] / phi[:,:,:,2])

plt.subplot(3,3,7)
plt.hist2d(x01[~np.isnan(x01*x02)], x02[~np.isnan(x01*x02)], bins=16)

plt.subplot(3,3,8)
plt.hist2d(x01[~np.isnan(x01*x12)], x12[~np.isnan(x01*x12)], bins=16)

plt.subplot(3,3,9)
plt.hist2d(x02[~np.isnan(x01*x12)], x12[~np.isnan(x01*x12)], bins=16)

plt.show()

