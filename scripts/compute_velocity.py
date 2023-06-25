import infotracking
from infotracking import Ensemble, infotheory
import numpy as np
import skimage
from skimage.io import imread,imsave
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Parameters -----------------------------------
path = '.'

startframe = 0
step = 1
nframes = 51
nt = nframes-1

windowsize = 64
windowspacing = 32
window_px0 = 0
window_py0 = 0

maxvel = 19

#------------------------------------------------
# Run analysis
#im = imread('../10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.tif')
# im = imread('C4-Pos12.25-40.tif')
#im = imread('../C4-Fused_12_13_14_15_1024.tif')
#im = imread('../10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8.ome.tif')
im = imread('../../../Microscopy/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.tif')
im = im[:,:,:,0]

#mask = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.mask.tif')
#mask = imread('C4-Pos12.25-40.mask.tif')
#mask = imread('../C4-Fused_12_13_14_15_1024.contour.mask.tif')
#mask = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8_phase.contour.mask.ome.tif')
#mask = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.contour.mask.ome.tif')
mask = imread('../../10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.contour.mask.tif')

#init_vel = np.load('vinit.npy')
init_vel = np.zeros(mask.shape + (2,))

#mask = np.zeros(im.shape[:3])
#x,y = np.meshgrid(np.arange(1024), np.arange(632))
#cx,cy = 320,500
#r = np.sqrt((x-cy)**2 + (y-cx)**2)
#m = r<300
#for frame in range(im.shape[0]):
#    mask[frame,:,:] = m

mask = mask / mask.max() # Make sure 0-1

im = im[startframe:startframe+(nframes * step):step,:,:]
mask = mask[startframe:startframe+(nframes * step):step,:,:]
#mask = mask[startframe-15:startframe-15+(nframes * step):step,:,:]

print("Image dimensions ",im.shape)
eg = Ensemble.EnsembleGrid(im, mask, init_vel, mask_threshold=0.5)

eg.initialise_ensembles(windowsize,windowsize, \
                        windowspacing,windowspacing, \
                        window_px0,window_py0)
print("Grid dimensions ", eg.gx,eg.gy)

eg.compute_motion(nt,maxvel,maxvel,velstd=21,dt=1)


# Generate some output
print("Saving quiver plots...")
eg.save_quivers(path, 'quiver_image_%04d.png', 'quiver_plain_%04d.png', normed=False)
print("Saving trajectory plots...")
eg.save_paths(path, 'path_image_%04d.png', 'path_plain_%04d.png')
print("Saving data files...")
eg.save_data(path)

