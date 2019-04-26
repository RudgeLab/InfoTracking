import infotracking
from infotracking import Ensemble, infotheory
import numpy as np
import skimage
from skimage.io import imread,imsave
from skimage import filters
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Parameters ------------------------------------
startframe = 0 
step = 1
path = sys.argv[1]
filename = os.path.join(path, sys.argv[2])
#maskfilename = os.path.join(path, sys.argv[3]) 
nframes = int(sys.argv[3])
nt = nframes-1

windowsize = 32 
windowspacing = 16
window_px0 = 0
window_py0 = 0

maxvel = 15 
#------------------------------------------------


# Run analysis
ima = [imread(filename%(startframe+(i)*step)).astype(np.float32) for i in range(nframes)]
ima = [filters.gaussian(im,1.0) for im in ima]
ima = np.array(ima)
print(ima.shape)
imb = [imread(filename%(startframe+(i)*step)).astype(np.float32) for i in range(nframes)]
imb = [filters.gaussian(im,1.0) for im in imb]
imb = np.array(imb)
print(imb.shape)
im = np.zeros((ima.shape[0],ima.shape[1],ima.shape[2],3), dtype=np.float32)
print("Image range", np.min(ima), np.max(ima))
im[:,:,:,0] = ima
im[:,:,:,1] = imb
#mask = ima > filters.threshold_triangle(ima) 
mask = np.ones(ima.shape).astype(np.float32) 
#np.array([imread(maskfilename%(0,startframe+(nframes-i)*step)).astype(np.float32) for i in range(nframes)])

print("Image dimensions ",im[0].shape)
eg = Ensemble.EnsembleGrid(im, mask)

eg.initialise_ensembles(windowsize,windowsize, \
                        windowspacing,windowspacing, \
                        window_px0,window_py0,
                        maxvel,maxvel)

print("Grid dimensions ", eg.gx,eg.gy)
print("Number of ensembles ", eg.n)

eg.compute_motion(nt,maxvel,maxvel,dt=1)

eg.save_quivers('quivers', 'quiver_%04d.png', 'quiver2_%04d.png', normed=False)
eg.save_paths('paths', 'path_%04d.png', 'path2_%04d.png')

eg.save_data('data')

eg.save_rois('rois', 'im1_%04d_%04d_%04d.tif', 'im2_%04d_%04d_%04d.tif')
eg.save_llmap('llmap', 'llmap_%04d_%04d_%04d.tif')

