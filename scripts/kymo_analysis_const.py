from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import distance_transform_edt
from scipy.signal import correlate
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.optimize import least_squares
from skimage.transform import warp_coords
from scipy.ndimage import map_coordinates
from scipy.signal import correlate, correlation_lags

start_frame = 0
end_frame = 120

im_all = imread('/Volumes/General/Microscopy/Conor Analysis/MG1655Z1/03-05-23_pAAA/Pos 9/10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.tif')
print(im_all.shape)
channels = [0,2,3]

mask_all = imread('/Volumes/General/Microscopy/Conor Analysis/MG1655Z1/03-05-23_pAAA/Pos 9/contour_masks/120_frames/10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.contour.mask.tif')
print(mask_all.shape)

nt = end_frame - start_frame

#x,y = np.meshgrid(np.arange(1200), np.arange(1200))
x,y = np.meshgrid(np.arange(1024), np.arange(1024))
nr = 64
kymo = np.zeros((nr,nt,3)) + np.nan
rsteps = np.zeros((nt,))
for t in range(nt):
    print(f'Processing frame {t}')
    im = im_all[start_frame + t,:,:,:].astype(float)
    mask = mask_all[start_frame + t,:,:].astype(float)

    for c in range(3):
        cim = im[:,:,channels[c]]
        cim = median_filter(cim, size=5)
        bg = cim[:100,:100].mean(axis=(0,1))
        cim = cim - bg
        cim[mask==0] = np.nan
        im[:,:,channels[c]] = cim

    #plt.subplot(1,4,1)
    #plt.imshow(im[:,:,0])
    #plt.colorbar()
    #plt.subplot(1,4,2)
    #plt.imshow(im[:,:,1])
    #plt.colorbar()
    #plt.subplot(1,4,3) 
    #plt.imshow(im[:,:,0] / im[:,:,1])
    #plt.colorbar()

    cx = x[mask>0].mean()
    cy = y[mask>0].mean()
    R = np.sqrt((x-cx)**2 + (y-cy)**2)

    edt = distance_transform_edt(mask)
    rmin = 0
    rmax = 512 # edt.max()
    rstep = (rmax - rmin) / nr
    rsteps[t] = rstep

    for r in range(nr):
        for c in range(3):
            imc = im[:,:,channels[c]]
            rimc = imc[(R>=(rmin + r*rstep))*(R<(rmin + (r+1)*rstep))]
            kymo[r,t,c] = np.mean(rimc)

np.save('kymo.npy', kymo)

#plt.figure(figsize=(12,5))
for c in range(3):
    plt.subplot(2,2,c+1)
    plt.imshow(kymo[:,:,c])
    plt.colorbar()

plt.savefig('kymos.png', dpi=300)
plt.show()

edge = np.zeros((nt,3)) + np.nan
for t in range(nt):
    for c in range(3):
        x = kymo[:,t,c]
        valid = ~np.isnan(x)
        idx = np.where(valid)[0]
        if len(idx)>1:
            idx_range = idx[-2:]
            edge[t,c] = x[idx_range].mean()

centre = kymo[:2,:,:].mean(axis=0)
plt.plot(edge)
plt.plot(centre)
plt.legend(['Edge', 'Centre'])
plt.show()

