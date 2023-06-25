import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt
from skimage import transform
from skimage.transform import downscale_local_mean
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.transform import warp_coords
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

im_all = imread('../Microscopy/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.tif')
#im_all = imread('../Microscopy/10x_1.0x_pLPT20_DHL_TiMain_1_MMStack_Pos5.ome.tif')
mask_all = imread('10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.contour.mask.tif')
#mask_all = imread('10x_1.0x_pLPT20_DHL_TiMain_1_MMStack_Pos5.ome.contour.mask.tif')
 

nt = 120

vfront = np.load('velocimetry/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0/vfront.npy')
mu0 = np.load('velocimetry/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0/mu0_fit.npy')
r0 = 50

snrkymo = np.load('snrkymo.npy')
plt.imshow(snrkymo)
plt.show()
for t in range(nt):
    m = mask_all[t,:,:]



