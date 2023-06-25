import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import fmin, least_squares
from scipy.signal import savgol_filter

nt = 144
step = 1

#mask_all = imread('C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8_phase.contour.mask.ome.tif')
mask_all = imread('../../10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.contour.mask.tif')
mask_all = mask_all>0
area = mask_all[:nt*step:step,:,:].sum(axis=(1,2))
radius = np.sqrt(area / np.pi)

rmax = np.zeros_like(radius)
for t in range(nt):
    m = mask_all[t*step,:,:]
    edt = distance_transform_edt(m)
    rmax[t] = edt.max()


#p = np.polyfit(np.arange(40), radius[10:50], 1)
#vfront = p[0]
#print(f'vfront = {vfront}')
vfront = savgol_filter(radius, 11, 3, deriv=1)
np.save('vfront.npy', vfront)
exprate = savgol_filter(area, 11, 3, deriv=1) / savgol_filter(area, 11, 3)
np.save('area_gr.npy', exprate)

plt.subplot(4,1,1)
plt.plot(area)
plt.subplot(4,1,2)
plt.plot(radius)
plt.subplot(4,1,3)
plt.plot(vfront)
plt.subplot(4,1,4)
plt.plot(exprate)
plt.show()

edt = np.zeros_like(mask_all).astype(float)
for t in range(nt):
    edt[t,:,:] = distance_transform_edt(mask_all[t,:,:])

r0 = 49.30114892335611
def fmin_func(t):
    def func(x):
        mu0 = x
        resid = []
        #D = r0 * (1 - np.exp(-rmax[t] / r0))
        #mu0 =  2 * vfront[t] / D
        gr = mu0 * np.exp(-edt[t,:,:]/r0)
        gr[edt[t,:,:]==0] = np.nan
        meangr = np.nanmean(gr, axis=(0,1))
        err = meangr - exprate[t]
        return err*err
    return func

mu0 = np.zeros((nt,))
for t in range(nt):
    func = fmin_func(t)
    res = fmin(func, x0=0.3)
    #res = least_squares(fmin_func(), x0=30, bounds=[0,rmax[50]])
    mu0[t] = res

np.save('mu0_fit.npy', mu0)
plt.plot(mu0)
plt.show()


