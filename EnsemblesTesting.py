import Ensemble
import numpy as np
import skimage
from skimage.io import imread,imsave
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

startframe = 600
step = 2
filename = sys.argv[1]
nframes = int(sys.argv[2])
nt = nframes-1
#filename = '/home/timrudge/cellmodeller/data/info_tracking-18-06-08-12-59/step-%05d.png'
#filename = '/home/timrudge/cellmodeller/data/random_grey-18-08-07-15-45/step-%05d.png'
#filename = '/media/timrudge/tjr34_ntfs/Microscopy/Cavendish/10.01.16/Pos0000/aligned_Frame%04dStep%04d.tiff'
#filename = '/media/timrudge/tjr34_ntfs/Microscopy/Cavendish/05.01.16/Pos0000/aligned_Frame%04dStep%04d.tiff'
#filename = '/home/timrudge/AndreaRavasioData/masked image/%02d.tif'
#filename = '/home/timrudge/ignacio_pickles/step-%05d.png'
#filename = '/home/timrudge/Code/InfoTracking/testdata/aligned_Frame%04dStep%04d.tif'
#filename = '/Users/timrudge/cellmodeller/data/weiner-18-08-01-17-44/step-%05d.png'
#filename = '/Users/timrudge/CavendishMicroscopy/10.01.16/Pos0000/Frame%d/aligned_Frame%04dStep%04d.tif'
ima = np.array([imread(filename%(startframe+(nframes-i)*step)).astype(np.float32) for i in range(nframes)])
ima = ima[:,0:-1:2,0:-1:2]
imb = np.array([imread(filename%(startframe+(nframes-i)*step)).astype(np.float32) for i in range(nframes)])
imb = imb[:,0:-1:2,0:-1:2]
im = np.zeros((ima.shape[0],ima.shape[1],ima.shape[2],3), dtype=np.float32)
im[:,:,:,0] = ima
im[:,:,:,1] = imb

print(im[0].shape)
plt.figure(figsize=(12,12))
plt.imshow(im[0]/2**16)
plt.colorbar()
eg = Ensemble.EnsembleGrid(im)
print(ima.shape)


eg.initialise_ensembles(32,32, 16,16, 8,8)
print(eg.gx,eg.gy)

eg.compute_motion(nt,7,7,dt=1)

eg.save_quivers('quivers', 'test_%04d.png', normed=True)
eg.save_paths('paths', 'test_%04d.png')

max_ll = eg.max_ll()
max_ll.shape
plt.imshow(-max_ll[:,:,0])
plt.colorbar()

fluo = eg.fluo()
print(fluo.shape)
plt.plot(fluo[10,16,:,0], fluo[10,16,:,1], '.')

eg.save_data('nparrays')
pos = np.fromfile('nparrays/pos.np')
pos.shape

#eg.save_rois('rois', 'test_%04d_%04d_%04d.tif')

