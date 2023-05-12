import numpy as np
from skimage.io import imread, imsave
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from skimage.morphology import flood
from skimage.morphology import area_closing
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from infotracking import Ensemble
import os
import matplotlib.pyplot as plt

# Functions 

def make_mask(weka, outfn):
    # Get number of frames
    nt,_,_ = weka.shape
    # Make binary mask of colony from weka output
    # Normalise
    mask = weka / weka.max()
    # Invert and make logical
    mask = mask==0
    for frame in range(nt):
        # Erode then dilate n times to remove small objects attached to colony
        n = 3
        for _ in range(n):
            mask[frame,:,:] = binary_erosion(mask[frame,:,:])
        for _ in range(n):
            mask[frame,:,:] = binary_dilation(mask[frame,:,:])

        # Flood fill to find central object (colony)
        mask[frame,:,:] = flood(mask[frame,:,:], (512,512))

        # Remove holes in colony
        mask[frame,:,:] = area_closing(mask[frame,:,:], area_threshold=1024)

    # Save final mask
    imsave(outfn, mask)

def compute_velocity(im, mask, windowsize, windowspacing, window_px0, window_py0, startframe, nframes, nt, step, maxvel,path):
    im = im[startframe:startframe+(nframes * step):step,:,:]
    mask = mask / mask.max() # Make sure 0-1
    mask = mask[startframe:startframe+(nframes * step):step,:,:]
    print("Image dimensions ",im.shape)

    # mask_threshold: percentage of the square that needs to be inside the colony. 100% in this case
    eg = Ensemble.EnsembleGrid(im, mask, mask_threshold=1)


    eg.initialise_ensembles(windowsize,windowsize, \
                            windowspacing,windowspacing, \
                            window_px0,window_py0)
    print("Grid dimensions ", eg.gx,eg.gy)

    eg.compute_motion(nt,maxvel,maxvel,velstd=5,dt=1)


    # Generate some output
    print("Saving quiver plots...")
    eg.save_quivers(path, 'quiver_image_%04d.png', 'quiver_plain_%04d.png', normed=False)
    print("Saving trajectory plots...")
    eg.save_paths(path, 'path_image_%04d.png', 'path_plain_%04d.png')
    print("Saving data files...")
    eg.save_data(path)


def process(startframe, step, im_all, mask_all, path):
    # Position and velocity arrays from velocimetry
    vel = np.load(os.path.join(path, 'vel.np.npy'))
    pos = np.load(os.path.join(path, 'pos.np.npy'))

    # Size of data
    nx,ny,nt,_ = vel.shape
    mask_all = mask_all / mask_all.max()

    # Make arrays to store results
    radpos = np.zeros((nt,nx,ny))
    vmag = np.zeros((nt,nx,ny))
    edt = np.zeros((nt,1024,1024))

    # Process the data and save results
    for frame in range(nt):
        print(f'Processing frame {frame}')

        # Subtract drift from velocities
        vel[:,:,frame,0] -= np.nanmean(vel[:,:,frame,0])
        vel[:,:,frame,1] -= np.nanmean(vel[:,:,frame,1])

        # Compute distance of each pixel from colony edge
        edt[frame,:,:] = distance_transform_edt(mask_all[startframe + frame*step,:,:])

        # Get direction to colony edge as negative of gradient of distance
        gx,gy = np.gradient(edt[frame,:,:])
        gx = -gx[pos[:,:,frame,0].astype(int)+31, pos[:,:,frame,1].astype(int)+31]
        gy = -gy[pos[:,:,frame,0].astype(int)+31, pos[:,:,frame,1].astype(int)+31]

        # Compute magnitude of velocities in radial direction
        vmag[frame,:,:] = vel[:,:,frame,0] * gx + vel[:,:,frame,1] * gy

        # Radial position of each grid square
        radpos[frame,:,:] = edt[frame, pos[:,:,frame,0].astype(int)+31, pos[:,:,frame,1].astype(int)+31]

    # Area and estimated radius of colony
    area = mask_all[startframe:startframe + nt*step:step,:,:].sum(axis=(1,2))
    radius = np.sqrt(area / np.pi)

    # Save results
    np.save(os.path.join(path, 'radpos.npy'), radpos)
    np.save(os.path.join(path, 'edt.npy'), edt)
    np.save(os.path.join(path, 'vmag.npy'), vmag)
    np.save(os.path.join(path, 'area.npy'), area)
    np.save(os.path.join(path, 'radius.npy'), radius)


def fit_velocity_profile(path):
    # Compute average growth rate of colony at each time point
    area = np.load(os.path.join(path, 'area.npy'))
    sarea = savgol_filter(area, 11, 3)
    dsarea = savgol_filter(area, 11, 3, deriv=1)
    mean_growth_rate = dsarea / area
    np.save(os.path.join(path, 'mean_growth_rate.npy'), mean_growth_rate)

    # Compute colony edge velocity
    radius = np.load(os.path.join(path, 'radius.npy'))
    vfront = savgol_filter(radius, 5, 3, deriv=1)
    np.save(os.path.join(path, 'vfront.npy'), vfront)

    # Normalize velocity by edge vel
    vmag = np.load(os.path.join(path, 'vmag.npy'))
    nt,_,_ = vmag.shape
    nvmag = vmag
    for frame in range(nt):
        nvmag[frame,:,:] = vmag[frame,:,:] / vfront[frame]

    # Fit an exponential decay model to the velocity data
    def residual_func(edt, nvmag, nt):
        def residuals(x):
            r0 = np.exp(x[0])
            C = x[1]
            res = []
            for frame in range(nt):
                for ix in range(32):
                    for iy in range(32):
                        if not np.isnan(nvmag[frame,ix,iy]):
                            r = edt[frame, ix*32:ix*32+64, iy*32:iy*32+64]
                            model_vmag = np.exp(-r/r0) + C
                            mean_model_vmag = model_vmag.mean()
                            res.append(mean_model_vmag - nvmag[frame, ix, iy])
            return res
        return residuals

    edt = np.load(os.path.join(path, 'edt.npy'))
    res = least_squares(residual_func(edt, nvmag, nt), x0=(np.log(50),0))
    r0 = np.exp(res.x[0])
    C = res.x[1]

    print(f'r0 = {r0}, C = {C}')

    # Make a plot to see how good the fit is
    radpos = np.load(os.path.join(path, 'radpos.npy'))
    x = radpos[~np.isnan(vmag)].ravel()
    y = np.zeros((0,))
    for frame in range(nt):
        for ix in range(32):
            for iy in range(32):
                if not np.isnan(nvmag[frame,ix,iy]):
                    r = edt[frame, ix*32:ix*32+64, iy*32:iy*32+64]
                    model_vmag = np.exp(-r/r0) + C
                    mean_model_vmag = model_vmag.mean()
                    y = np.append(y, mean_model_vmag)

    plt.plot(x, nvmag[~np.isnan(nvmag)], 'x')
    plt.plot(x, y, '.')
    plt.savefig(os.path.join(path, f'velocity_fit_r0 = {r0}, C = {C}.png'))
    plt.clf()


# Parameters -----------------------------------

# Positions to be analyzed
col_pos = [5, 6, 7, 8, 9]
path_data = os.path.join(".", "Data")


for col in col_pos:
    # create a folder for storing the resulting data
    path = os.path.join(".", "Results", f"Pos{col}")
    os.mkdir(path)
    # File names
    imfn = os.path.join(path_data, f"10x_1.5x_-5_pAAA_MG_1_MMStack_Pos{col}.ome.phase.tif")
    wekafn = os.path.join(path_data, f"10x_1.5x_-5_pAAA_MG_1_MMStack_Pos{col}.ome.weka.tif")
    mask_outfn = os.path.join(path_data, f"10x_1.5x_-5_pAAA_MG_1_MMStack_Pos{col}.ome.mask.tif")

    # Read files
    weka = imread(wekafn)
    im = imread(imfn)

    # for compute_velocity
    startframe = 10
    step = 1
    nframes = 100
    nt = nframes-1

    # window size of square used for computing velocity
    windowsize = 64
    # distance between windows, so they overlap
    windowspacing = 32
    window_px0 = 0
    window_py0 = 0

    # velocity threshold
    maxvel = 7

    #make_mask(weka, mask_outfn)
    mask = imread(mask_outfn)
    compute_velocity(im, mask, windowsize, windowspacing, window_px0, window_py0, startframe, nframes, nt, step, maxvel,path)
    process(startframe, step, im, mask, path)
    fit_velocity_profile(path)