import numpy as np
from scipy.ndimage import fourier_shift
from scipy import ndimage
from scipy.ndimage.filters import laplace as laplace
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom
from skimage.feature import register_translation
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import infotheory
import os
import sys


path = sys.argv[1]
#'/media/timrudge/tjr34_ntfs/Microscopy/Cavendish/10.01.16/Pos0000'
infile = os.path.join(path, 'Frame%04dStep%04d.tiff')
outfile = os.path.join(path, 'aligned_Frame%04dStep%04d.tiff')
startframe = 0
step = 1
nframes = 250

im1 = imread(infile%(0,startframe)).astype(np.float32) 
im1 = gaussian_filter(im1,1)
im2 = imread(infile%(0,startframe+step)).astype(np.float32) 
im2 = gaussian_filter(im2,1)


shifts = np.zeros((nframes-step,2))

# Offset images to align to sequentially
for i in range(nframes-1):
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

    # Reset reference image and get next image in sequence
    im1 = offset_image
    im2 = imread(infile%(0,startframe+(i+2)*step)).astype(np.float32) 
    im2 = gaussian_filter(im2,1)

# Shift 2nd channel by same offset 
for i in range(nframes-1):
    im = imread(infile%(2,startframe+(i+1)*step)).astype(np.float32) 
    im = gaussian_filter(im,1)
    # Pad image to avoid periodic wrap around
    pim = np.zeros((im.shape[0]+400,im2.shape[1]+400))
    pim[200:-200,200:-200] = im
    # Shift image in fourier space
    offset_image = fourier_shift(np.fft.fftn(pim), -shifts[i])
    offset_image = np.fft.ifftn(offset_image).real
    # Remove padding
    offset_image = offset_image[200:-200,200:-200]
    imsave(outfile%(2,(i+1)*step), offset_image.astype(np.uint16), plugin='tifffile')

