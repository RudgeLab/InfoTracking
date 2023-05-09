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

startframe = 70
step = 1
nframes = 30
nt = nframes-1

windowsize = 64 
windowspacing = 32
window_px0 = 0
window_py0 = 0

maxvel = 7

#------------------------------------------------
# Run analysis
im = imread('../10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.tif')
im = im[startframe:startframe+(nframes * step):step,:,:,1]
mask = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.mask.tif')
mask = mask / mask.max() # Make sure 0-1
mask = mask[startframe:startframe+(nframes * step):step,:,:]

print("Image dimensions ",im.shape)
eg = Ensemble.EnsembleGrid(im, mask, mask_threshold=1)

eg.initialise_ensembles(windowsize,windowsize, \
                        windowspacing,windowspacing, \
                        window_px0,window_py0)
print("Grid dimensions ", eg.gx,eg.gy)

eg.compute_motion(nt,maxvel,maxvel,velstd=5,dt=1)


# Generate some output
print("Saving quiver plots...")
eg.save_quivers(path, 'quiver_image_%04d.png', 'quiver_plain_%04d.png', normed=False)
print("Saving trajectory plots...")
eg.save_paths(path, 'path_image_%04d.png', 'path_plain_%04d.png')
print("Saving data files...")
eg.save_data(path)

