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

startframe = 1
step = 1
nframes = 40
nt = nframes-1

windowsize = 64 
windowspacing = 16
window_px0 = 8
window_py0 = 8

maxvel = 15
#------------------------------------------------

# File names and paths
impath = os.path.join(path, 'masked image')
mskpath = os.path.join(path, 'mask')
filename = os.path.join(impath, '%d.tif')
maskfilename = os.path.join(mskpath, '%d.tif') 

#------------------------------------------------
# Run analysis
im = np.array([imread(filename%(startframe+i*step)).astype(np.float32) for i in range(nframes)])
mask = np.array([imread(maskfilename%(startframe+i*step)).astype(np.float32) for i in range(nframes)])

print("Image dimensions ",im[0].shape)
eg = Ensemble.EnsembleGrid(im, mask)

eg.initialise_ensembles(windowsize,windowsize, \
                        windowspacing,windowspacing, \
                        window_px0,window_py0)
print("Grid dimensions ", eg.gx,eg.gy)

eg.compute_motion(nt,maxvel,maxvel,dt=1)


# Generate some output
print("Saving quiver plots...")
eg.save_quivers(path, 'quiver_image_%04d.png', 'quiver_plain_%04d.png', normed=True)
print("Saving trajectory plots...")
eg.save_paths(path, 'path_image_%04d.png', 'path_plain_%04d.png')
print("Saving data files...")
eg.save_data(path)

