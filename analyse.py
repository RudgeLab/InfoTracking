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

# Parameters ------------------------------------
startframe = 100
step = 1
path = sys.argv[1]
filename = os.path.join(path, sys.argv[2])
maskfilename = os.path.join(path, sys.argv[3]) 
nframes = int(sys.argv[4])
nt = nframes-1

windowsize = 32 
windowspacing = 32
window_px0 = 0
window_py0 = 0

maxvel = 15
#------------------------------------------------

# Run analysis
ima = np.array([imread(filename%(0,startframe+(nframes-i)*step)).astype(np.float32) for i in range(nframes)])
imb = np.array([imread(filename%(2,startframe+(nframes-i)*step)).astype(np.float32) for i in range(nframes)])
im = np.zeros((ima.shape[0],ima.shape[1],ima.shape[2],3), dtype=np.float32)
im[:,:,:,0] = ima
im[:,:,:,1] = imb
mask = np.array([imread(maskfilename%(0,startframe+(nframes-i)*step)).astype(np.float32) for i in range(nframes)])

print("Image dimensions ",im[0].shape)
eg = Ensemble.EnsembleGrid(im, mask)

eg.initialise_ensembles(windowsize,windowsize, \
                        windowspacing,windowspacing, \
                        window_px0,window_py0)
print("Grid dimensions ", eg.gx,eg.gy)

eg.compute_motion(nt,maxvel,maxvel,dt=1)

eg.save_quivers(path, 'quiver_%04d.png', normed=True)
eg.save_paths(path, 'path_%04d.png')

eg.save_data(path)

#eg.save_rois('rois', 'test_%04d_%04d_%04d.tif')

