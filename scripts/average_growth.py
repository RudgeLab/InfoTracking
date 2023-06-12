import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import fmin

nt = 144
step = 1

mask_all = imread('C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8_phase.contour.mask.ome.tif')
mask_all = mask_all>0
area = mask_all[:nt*step:step,:,:].sum(axis=(1,2))
radius = np.sqrt(area / np.pi)

rmax = np.zeros_like(radius)
for t in range(nt):
    m = mask_all[t*step,:,:]
    edt = distance_transform_edt(m)
    rmax[t] = edt.max()


p = np.polyfit(np.arange(40), radius[10:50], 1)
vfront = p[0]
print(f'vfront = {vfront}')

edt = np.zeros_like(mask_all).astype(float)
for t in range(nt):
    edt[t,:,:] = distance_transform_edt(mask_all[t,:,:])

def func(x):
    gr = (2 * vfront / x) * np.exp(-edt/x)
    gr[edt==0] = np.nan
    meangr = np.nanmean(gr, axis=(1,2))
    exprate = np.diff(area) / area[:-1]
    err = meangr[:-1] - exprate
    return np.sum(err*err)

res = fmin(func, 30)
r0 = res[0]
print(f'r0 = {r0}')

plt.subplot(4,1,1)
plt.plot(area)
plt.subplot(4,1,2)
plt.plot(radius)
plt.subplot(4,1,3)
plt.plot(rmax)
plt.subplot(4,1,4)
plt.plot(np.diff(area)/area[:-1])
plt.show()

