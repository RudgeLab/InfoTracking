import numpy as np
import numpy.matlib
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt
from skimage import transform
from skimage.transform import downscale_local_mean
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.transform import warp_coords, rescale
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

start_frame = 15
step = 1

nt = 25 

# Load the tif image from the microscope
#fname = '10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8.ome.tif'
fname = 'Fused_12_13_14_15_1024.bgcorr.tiff'
im_all = imread('../' + fname)
im_all = im_all[:,20:-20,20:-20,:]
im_all = im_all.astype(float)
print(im_all.shape)
channels = [0,1,2]

#mask_all = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8_phase.mask.ome.tif')
#mask_all = imread('mask_contour.tif')
mask_all = imread('../C4-Fused_12_13_14_15_1024.contour.mask.tif')
mask_all = mask_all>0

w,h = im_all.shape[1:3]
print(w,h)
x,y = np.meshgrid(np.arange(h), np.arange(w))
#cx,cy = 512,512

nr = 16
na = 32
astep = 2 * np.pi / na
#crad = np.sqrt((x-cy)**2 + (y-cx)**2)
polar = np.zeros((nt,nr,na,3))
for t in range(nt):
    m = mask_all[t*step,:,:]
    print(m.shape, y.shape)
    cx = y[m].mean()
    cy = x[m].mean()
    ang = np.arctan2(x-cy, y-cx)
    ang[ang<0] = ang[ang<0] + 2*np.pi
    edt = distance_transform_edt(m)
    rmax = edt.max()
    rstep = rmax/nr
    for r in range(nr):
        for a in range(na):
            idx = (edt>=r*rstep) * (edt<(r+1)*rstep) * (ang>=a*astep) * (ang<(a+1)*astep)
            for c in range(3):
                cim = im_all[start_frame + step*t,:,:,channels[c]]
                fr = cim[idx]
                polar[t,r,a,c] = fr.mean()
    
    polar_im = rescale(polar[t,:,:,:], (rstep,astep*180/np.pi,1))
    np.save('osc_polar_im_t_%03d.npy'%t, polar_im)


np.save('osc_polar.npy', polar)

for t in range(nt):
    for c in range(3):
        tcim = polar[t,:,:,c]
        polar[t,:,:,c] = tcim / np.nanmax(tcim)

np.save('osc_npolar.npy', polar)


