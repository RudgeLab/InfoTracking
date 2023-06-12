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

mask = imread('../C4-Fused_12_13_14_15_1024.contour.mask.tif')
mask = mask>0

fname = 'Fused_12_13_14_15_1024.bgcorr.tiff'
im_all = imread('../' + fname)
im_all = im_all[:,20:-20,20:-20,:]
im_all = im_all.astype(float) 

nt = 25

vfront = 8.636534422186193
mu0 = 0.22696434875613344
r0 = 76.10476685
v = np.zeros(mask.shape + (2,))

for t in range(nt-1):
    m = mask[t,:,:]
    edt = distance_transform_edt(m)

    vmag = vfront * np.exp(-edt/r0)

    # Get direction to colony edge as negative of gradient of distance
    gx,gy = np.gradient(edt)
 
    v[t,:,:,0] = -vmag * gx
    v[t,:,:,1] = -vmag * gy
    
np.save('vinit.npy', v)


