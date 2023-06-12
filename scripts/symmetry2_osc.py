import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt
from skimage import transform
from skimage.transform import downscale_local_mean
from skimage.metrics import mean_squared_error, normalized_root_mse
import matplotlib.pyplot as plt

start_frame = 90
step = 1

nt = 30

# Load the tif image from the microscope
fname = 'Fused_12_13_14_15_1024.bgcorr.tiff'
im_all = imread('../' + fname)
im_all = im_all.astype(float)
print(im_all.shape)
channels = [0,1,2]

mask_all = np.zeros(im_all.shape[:3])
x,y = np.meshgrid(np.arange(1024), np.arange(632))
cx,cy = 320,500
r = np.sqrt((x-cy)**2 + (y-cx)**2)
mask = r<300
for frame in range(im_all.shape[0]):
    mask_all[frame,:,:] = mask


nr = 16
na = 32
astep = 2 * np.pi / na
polar = np.zeros((nt,nr,na,3))
for t in range(nt):
    edt = distance_transform_edt(mask_all[start_frame + t*step,:,:])
    rmax = edt.max()
    rstep = rmax/nr
    for r in range(nr):
        ang = np.arctan2(x-cy, y-cx)
        ang[ang<0] = ang[ang<0] + 2*np.pi
        for a in range(na):
            idx = (edt>=r*rstep) * (edt<(r+1)*rstep) * (ang>=a*astep) * (ang<(a+1)*astep)
            for c in range(3):
                cim = im_all[start_frame + step*t,:,:,channels[c]]
                fr = cim[idx]
                polar[t,r,a,c] = fr.mean()

np.save('polar_osc.npy', polar)

for t in range(nt):
    for c in range(3):
        tcim = polar[t,:,:,c]
        polar[t,:,:,c] = tcim / tcim.max()

np.save('npolar_osc.npy', polar)

