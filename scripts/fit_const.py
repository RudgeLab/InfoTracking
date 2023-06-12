import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

im = imread('C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.tif')
im = im[50:,:,:,:3]

mask = imread('C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.mask.tif')
mask = mask[50:,:,:]



nt,nx,ny = np.shape(mask)

for t in range(nt):
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
dssim = np.zeros((nt,40,64,3))
dsdsim = np.zeros((nt,40,64,3))
for t in range(nt):
    for c in range(3):
        dssim[t,:,:,c] = downscale_local_mean(sim[t,:,:,c], 16)
        dsdsim[t,:,:,c] = downscale_local_mean(dsim[t,:,:,c], 16)

sim = dssim[:nframes,:,:,:]
dsim = dsdsim[:nframes,:,:,:]
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

