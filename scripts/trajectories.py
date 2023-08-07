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

im_all = imread('/Volumes/General//Microscopy/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.ome.tif')
#im_all = imread('../Microscopy/10x_1.0x_pLPT20_DHL_TiMain_1_MMStack_Pos5.ome.tif')
im_all = im_all[:,:,:,[2,3]]
mask_all = imread('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.contour.mask.ome.tif')
#mask_all = imread('10x_1.0x_pLPT20_DHL_TiMain_1_MMStack_Pos5.ome.contour.mask.tif')
 

nt = 125
rmax = 64
centre_prof = np.zeros((nt,2))
edge_prof = np.zeros((nt,2))
for t in range(nt):
    m = mask_all[t,:,:]
    edt = distance_transform_edt(m)
    idx_edge = (edt < rmax) * (edt > 0)
    idx_centre = (edt > (edt.max() - rmax)) * (edt>0)
    for c in range(2):
        bg = im_all[0,:,:,c][m==0].mean()
        cim = im_all[t,:,:,c] - bg
        edge_prof[t,c] = cim[idx_edge].mean()
        centre_prof[t,c] = cim[idx_centre].mean()

#for c in range(2):
#    edge_prof[:,c] = edge_prof[:,c] - edge_prof[:,c].min()
#    centre_prof[:,c] = centre_prof[:,c] - centre_prof[:,c].min()

plt.subplot(1,2,1)
plt.plot(edge_prof[:,0])
plt.plot(centre_prof[:,0])
plt.legend(['Edge', 'Centre'])

plt.subplot(1,2,2)
plt.plot(edge_prof[:,1])
plt.plot(centre_prof[:,1])
plt.legend(['Edge', 'Centre'])
plt.show()

np.save('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.centre.profile.npy', centre_prof)
np.save('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.edge.profile.npy', edge_prof)

# Compute expression rates
#deg_rate = np.log(2)/(28/10) + np.log(2)/(10/10)
sep = np.zeros_like(edge_prof)
dsep = np.zeros_like(edge_prof)
scp = np.zeros_like(centre_prof)
dscp = np.zeros_like(centre_prof)
for c in range(2):
    sep[:,c] = savgol_filter(edge_prof[:,c], 21, 3)
    dsep[:,c] = savgol_filter(edge_prof[:,c], 21, 3, deriv=1)
    scp[:,c] = savgol_filter(centre_prof[:,c], 21, 3)
    dscp[:,c] = savgol_filter(centre_prof[:,c], 21, 3, deriv=1)

np.save('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.edge.sprofile.npy', sep)
np.save('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.edge.dsprofile.npy', dsep)
np.save('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.centre.sprofile.npy', scp)
np.save('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.centre.dsprofile.npy', dscp)





