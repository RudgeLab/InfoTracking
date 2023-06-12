import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt
from skimage import transform
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt

start_frame = 87
step = 1

nt = 1

# Load the tif image from the microscope
fname = 'Fused_12_13_14_15_1024.bgcorr.tiff' # '10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.tif'
im_all = imread('../' + fname)
im_all = im_all.astype(float)
print(im_all.shape)
channels = [0,1,2]

#mask = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.mask.tif')
#mask = mask>0

mask_all = np.zeros(im_all.shape[:3])
x,y = np.meshgrid(np.arange(1024), np.arange(632))
cx,cy = 320,500
r = np.sqrt((x-cy)**2 + (y-cx)**2)
mask = r<300
for frame in range(im_all.shape[0]):
    mask_all[frame,:,:] = mask

cx,cy = 320,500

ntheta = 32
corr = np.zeros((nt, ntheta, 3))
for t in range(nt):
    #x,y = np.meshgrid(np.arange(1024), np.arange(1024))
    #cx = np.nanmean(x[mask[start_frame+t*step,:,:]])
    #cy = np.nanmean(y[mask[start_frame+t*step,:,:]])
    for theta in range(ntheta):
        ang = 2*np.pi*theta/ntheta - np.pi
        ref_im = np.zeros((40,64,3))
        transformed_im = np.zeros((40,64,3))
        for c in range(3):
            im = im_all[start_frame + step*t,:,:,channels[c]]
            msk = mask_all[start_frame + step*t,:,:]
            im[msk==0] = np.nan
            dsim = downscale_local_mean(im, 16)

            #tform = transform.EuclideanTransform(
            #        rotation = 0,
            #        translation = (316-cx,512-cy)
            #        )
            #tim = transform.warp(im, tform.inverse)
            dsim = dsim / np.nanmax(dsim)
            ref_im[:,:,c] = dsim

            tf_rotate = transform.EuclideanTransform(rotation=ang)
            tf_shift = transform.EuclideanTransform(translation=[-cy, -cx])
            tf_shift_inv = transform.EuclideanTransform(translation=[cy, cx])

            rtim = transform.warp(im, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
            rmsk = transform.warp(msk, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)

            #frtim = rtim[::-1,:]

            #tf_rotate2 = transform.EuclideanTransform(rotation=-ang)
            #rfrtim = transform.warp(frtim, (tf_shift + (tf_rotate2 + tf_shift_inv)).inverse)

            rtim[rmsk==0] = np.nan
            dsrtim = downscale_local_mean(rtim, 16)
            dsrtim = dsrtim / np.nanmax(dsrtim)
            transformed_im[:,:,c] =  dsrtim # rfrtim

            #idx = ~np.isnan(dsrtim) * ~np.isnan(dsim)
            idx = np.isnan(dsrtim) + np.isnan(dsim)
            dsrtim[idx] = 0
            dsim[idx] = 0
            #x = dsrtim[idx].ravel()
            #y = dsim[idx].ravel()
            #correlation = np.corrcoef(x, y)
            corr[t,theta,c] = normalized_root_mse(dsrtim, dsim)
            print(corr[t,theta,c])

        '''
        idx = ~np.isnan(transformed_im) * ~np.isnan(ref_im)
        x = transformed_im[idx].ravel()
        y = ref_im[idx].ravel()
        correlation = np.corrcoef(x, y)
        print(correlation)
        transformed_im[np.isnan(transformed_im)] = 0
        ref_im[np.isnan(ref_im)] = 0
        mse = normalized_root_mse(transformed_im, ref_im) # correlation[0,1]
        corr[t,theta] = correlation[0,1]
        print(mse)
        '''

np.save('corr.npy', corr)
