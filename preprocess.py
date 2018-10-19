import numpy as np
from scipy.ndimage import fourier_shift
from scipy import ndimage
from scipy.ndimage.filters import laplace as laplace
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter

from skimage.feature import register_translation
from skimage.transform import downscale_local_mean
from skimage.io import imread, imsave

import matplotlib.pyplot as plt
import infotracking
from infotracking import infotheory
import os
import sys

###
# Preprocess timelapse images
# 1. Gaussian smoothing
# 2. Subsample 2x
# 3. Compute and remove background
# 4. Register
###

path = sys.argv[1]
nframes = int(sys.argv[2])
#'/media/timrudge/tjr34_ntfs/Microscopy/Cavendish/10.01.16/Pos0000'
infile = os.path.join(path, 'Frame%04dStep%04d.tif')
outfile = os.path.join(path, 'aligned_Frame%04dStep%04d.tiff')
outfile_mask = os.path.join(path, 'aligned_mask_Frame%04dStep%04d.tiff')
startframe = 0
step = 1
scale = 2 

im1 = imread(infile%(0,startframe), plugin='tifffile').astype(np.float32) 
im1 = gaussian_filter(im1,1)
im1 = downscale_local_mean(im1, (scale,scale))

# Use first image to compute background of 1st channel
bgmean = np.mean(im1.ravel())
bgstd = np.std(im1.ravel())
bgval = bgmean + bgstd*2
print(bgmean)

im2 = imread(infile%(2,startframe+step), plugin='tifffile').astype(np.float32) 
im2 = gaussian_filter(im2,1)
im2 = downscale_local_mean(im2, (scale,scale))


shifts = np.zeros((nframes-step,2))

# Offset images to align to sequentially
for i in range(nframes-1):
    # Subtract background
    im1 = im1-bgval
    im1[im1<0] = 0
    im2 = im2-bgval
    im2[im2<0] = 0

    # Register images
    shift,_,_ = register_translation(im2,im1)
    print(i, shift)
    shifts[i] = shift

    # Pad image to avoid periodic wrap around
    pim2 = np.zeros((im2.shape[0]+400,im2.shape[1]+400))
    pim2[200:-200,200:-200] = im2

    # Shift image in fourier space
    offset_image = fourier_shift(np.fft.fftn(pim2), -shift)
    offset_image = np.fft.ifftn(offset_image).real
    # Remove padding
    offset_image = offset_image[200:-200,200:-200]
    
    imsave(outfile%(0,i+1), offset_image.astype(np.uint16), plugin='tifffile')

    immask = offset_image
    immask = gaussian_filter(immask,10)
    mask = immask > bgval
    imsave(outfile_mask%(0,i+1), mask.astype(np.uint16), plugin='tifffile')

    # Reset reference image and get next image in sequence
    im1 = offset_image
    im2 = imread(infile%(0,startframe+(i+2)*step), plugin='tifffile').astype(np.float32) 
    im2 = gaussian_filter(im2,1)
    im2 = downscale_local_mean(im2, (scale,scale))

# Shift 2nd channel by same offset 

# Use first image to compute background of 2nd channel
im = imread(infile%(2,startframe)).astype(np.float32) 
im = gaussian_filter(im,1)
im = downscale_local_mean(im, (scale,scale))
bgmean = np.mean(im.ravel())
bgstd = np.std(im.ravel())
bgval = bgmean + bgstd*2

for i in range(nframes-1):
    im = imread(infile%(2,startframe+(i+1)*step)).astype(np.float32) 
    im = gaussian_filter(im,1)
    im = downscale_local_mean(im, (scale,scale))
    # Subtract background
    im = im-bgval
    im[im<0] = 0
    
    # Pad image to avoid periodic wrap around
    pim = np.zeros((im.shape[0]+400,im2.shape[1]+400))
    pim[200:-200,200:-200] = im
    # Shift image in fourier space
    offset_image = fourier_shift(np.fft.fftn(pim), -shifts[i])
    offset_image = np.fft.ifftn(offset_image).real
    # Remove padding
    offset_image = offset_image[200:-200,200:-200]
    imsave(outfile%(2,(i+1)*step), offset_image.astype(np.uint16), plugin='tifffile')

