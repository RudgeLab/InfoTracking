import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

im = imread('Fused_12_13_14_15_1024.bgcorr.tiff')
im = im[87:,:,:,:3]

x,y = np.meshgrid(np.arange(1024), np.arange(632))
cx,cy = 320,500
r = np.sqrt((x-cy)**2 + (y-cx)**2)
mask = r<300

for t in range(57):
    for c in range(3):
        ctim = im[t,:,:,c]
        ctim[~mask] = np.nan

sim = np.zeros_like(im) + np.nan
dsim = np.zeros_like(im) + np.nan
for ix in range(632):
    for iy in range(1024):
        if mask[ix,iy]:
            for c in range(3):
                sim[:,ix,iy,c] = savgol_filter(im[:,ix,iy,c], 15, 3)
                dsim[:,ix,iy,c] = savgol_filter(im[:,ix,iy,c], 15, 3, deriv=1)

np.save('sim.npy', sim)
np.save('dsim.npy', dsim)

nframes = 10

sim = np.load('sim.npy')
dsim = np.load('dsim.npy')

from skimage.transform import downscale_local_mean
dssim = np.zeros((57,40,64,3))
dsdsim = np.zeros((57,40,64,3))
for t in range(57):
    for c in range(3):
        dssim[t,:,:,c] = downscale_local_mean(sim[t,:,:,c], 16)
        dsdsim[t,:,:,c] = downscale_local_mean(dsim[t,:,:,c], 16)

sim = dssim[:nframes*4:4,:,:,:]
dsim = dsdsim[:nframes*4:4,:,:,:]
np.save('dssim.npy', sim)
np.save('dsdsim.npy', dsim)

halflife = 2 # 2x10=20 minute protein halflife
gamma = np.log(2) / halflife
phi = dsim + sim*gamma
np.save('phi.npy', phi)

rphi = dsim / sim
np.save('rphi.npy', rphi)

phi01 = phi[:,:,:,0] / phi[:,:,:,1]
phi02 = phi[:,:,:,0] / phi[:,:,:,2]
phi12 = phi[:,:,:,1] / phi[:,:,:,2]
np.save('phi01.npy', phi01)
np.save('phi02.npy', phi02)
np.save('phi12.npy', phi12)

'''
plt.subplot(1,3,1)
plt.imshow(phi01[0,:,:])
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(phi02[0,:,:])
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(phi12[0,:,:])
plt.colorbar()
plt.show()
'''


'''

def res_func(phi01, phi02, phi12, n, eps):
    def func(x):
        a0,a1,a2 = np.exp(x[:3])
        p0 = np.exp(x[3:n+3])
        p1 = np.exp(x[n+3:2*n+3])
        p2 = np.exp(x[2*n+3:3*n+3])
        r0 = a0 / (1 + p1)
        r1 = a1 / (1 + p2)
        r2 = a2 / (1 + p0)
        r01 = phi01 - r0/r1
        r02 = phi02 - r0/r2
        r12 = phi12 - r1/r2
        residuals = np.concatenate((r01, r02, r12, eps * x))
        return residuals
    return func

n = len(phi01[~np.isnan(phi01)])
print(n)
x0 = np.zeros((3 + n*3,))
x0[:3] = 100
x0[3:] = 1
bounds = [[-10,-10,-10]+[-10]*n*3, [10,10,10]+[0]*n*3]
res = least_squares(res_func(phi01[~np.isnan(phi01)], phi02[~np.isnan(phi02)], phi12[~np.isnan(phi12)], n=n, eps=0), x0=np.log(x0))
x = res.x
a = np.exp(x[:3])
p0 = np.exp(x[3:n+3])
p1 = np.exp(x[n+3:2*n+3])
p2 = np.exp(x[2*n+3:3*n+3])

mask = ~np.isnan(phi01)
p0map = np.zeros_like(phi01) + np.nan
p0map[mask] = p0
p1map = np.zeros_like(phi01) + np.nan
p1map[mask] = p1
p2map = np.zeros_like(phi01) + np.nan
p2map[mask] = p2

pmap = np.zeros(p0map.shape + (3,))
pmap[:,:,:,0] = 1/(1+p1map)
pmap[:,:,:,1] = 1/(1+p2map)
pmap[:,:,:,2] = 1/(1+p0map)

np.save('pmap.npy', pmap)
np.save('a.npy', a)
print(a)

'''
