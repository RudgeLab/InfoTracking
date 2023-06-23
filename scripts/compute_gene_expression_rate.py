import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

start_frame = 40
step = 1

# Load the velocity field
vel = np.load('vel.np.npy')
nx,ny,nt,_ = vel.shape
for t in range(nt):
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            vel[ix,iy,t,0] = np.nanmedian(vel[ix-2:ix+3,iy-2:iy+3,t,0])
            vel[ix,iy,t,1] = np.nanmedian(vel[ix-2:ix+3,iy-2:iy+3,t,1])
#    vel[:,:,t,0] = median_filter(vel[:,:,t,0], size=5)
#    vel[:,:,t,1] = median_filter(vel[:,:,t,1], size=5)

# Load the radial velocity magnitude
vmag = np.load('vmag.npy')

# Load the tif image from the microscope
#im_all = imread('../10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8.ome.tif').astype(float)
im_all = imread('../10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.tif').astype(float)
print(im_all.shape)
channels = [0,2,3]

#mask_all = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos8_phase.contour.mask.ome.tif')
mask_all = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.contour.mask.ome.tif')

im_all[mask_all==0] = np.nan

# Compute fluorescence of each grid square
fluo0 = np.zeros((nt, 64, 64, 3)) + np.nan
fluo1 = np.zeros((nt, 64, 64, 3)) + np.nan
for frame in range(nt):
    for ix in range(64):
        for iy in range(64):
            if not np.isnan(vmag[frame,ix,iy]):
                vx,vy = vel[ix,iy,frame,:].astype(int)
                f0 = im_all[start_frame+frame*step, ix*16:ix*16+64, iy*16:iy*16+64,[0,2,3]]
                f1 = im_all[start_frame+(frame+1)*step, ix*16+vx:ix*16+64+vx, iy*16+vy:iy*16+64+vy,[0,2,3]]
                fluo0[frame,ix,iy,:] = np.nanmean(f0, axis=(1,2))
                fluo1[frame,ix,iy,:] = np.nanmean(f1, axis=(1,2))
dfluo = fluo1 - fluo0

# Compute radial average of fluorescence and derivative, df/ft
radpos = np.load('radpos.npy')
rmax = radpos.max()
nr = 20
rstep = rmax/nr
rfluo = np.zeros((nt,nr, 3))
drfluo = np.zeros((nt,nr, 3))
rfluo_var = np.zeros((nt,nr, 3))
drfluo_var = np.zeros((nt,nr, 3))
for frame in range(nt):
    for c in range(3):
        f = fluo0[frame,:,:,c]
        df = dfluo[frame,:,:,c]
        for r in range(nr):
            rfluo[frame,r,c] = np.nanmean(f[(radpos[frame,:,:]>=r*rstep)*(radpos[frame,:,:]<(r+1)*rstep)])
            drfluo[frame,r,c] = np.nanmean(df[(radpos[frame,:,:]>=r*rstep)*(radpos[frame,:,:]<(r+1)*rstep)])
            rfluo_var[frame,r,c] = np.nanvar(f[(radpos[frame,:,:]>=r*rstep)*(radpos[frame,:,:]<(r+1)*rstep)])
            drfluo_var[frame,r,c] = np.nanvar(df[(radpos[frame,:,:]>=r*rstep)*(radpos[frame,:,:]<(r+1)*rstep)])

clim0 = [np.nanmin(dfluo[:,:,:,0]), np.nanmax(dfluo[:,:,:,0])]
clim1 = [np.nanmin(dfluo[:,:,:,1]), np.nanmax(dfluo[:,:,:,1])]
clim2 = [np.nanmin(dfluo[:,:,:,2]), np.nanmax(dfluo[:,:,:,2])]
for t in range(9):
    plt.subplot(1,3,1)
    plt.imshow(dfluo[t,:,:,0], clim=clim0)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(dfluo[t,:,:,1], clim=clim1)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(dfluo[t,:,:,2], clim=clim2)
    plt.colorbar()
    plt.show()

'''
# Smooth df/ft and remove nans
from scipy.interpolate import interp1d
for r in range(nr):
    for c in range(3):
        x = np.arange(nt)
        y = drfluo[:,r,c]
        if len(x[~np.isnan(y)])>0:
            iy = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], bounds_error=False, fill_value='extrapolate')
            sy = savgol_filter(iy(x), 11, 3)
            drfluo[~np.isnan(y),r,c] = sy[x[~np.isnan(y)]]

# Compute expression rate (kymograph)
vfront = np.load('vfront.npy')
r = np.arange(0, 20) * rstep
r0 = 30
mu0 = 2 * vfront / r0 
rer = np.zeros((nt,nr, 3)) + np.nan
for frame in range(nt):        
    for c in range(3):
        f = rfluo[frame,:,c]
        dfdt = drfluo[frame,:,c]
        dfdt = dfdt[~np.isnan(f)]
        rr = r[~np.isnan(f)]
        ff = f[~np.isnan(f)]
        if len(ff)>5:
            dfdr = savgol_filter(ff, 5, 3, deriv=1) / rstep
            drdt = vfront[frame] * np.exp(-rr/r0)
            gr = mu0[frame] * np.exp(-rr/r0)
            rer[frame,~np.isnan(f),c] = dfdt + dfdr*drdt + (gr + 0.05)*ff

# Remap kymograph from r relative to edge, to r relative to centre
rrer = np.zeros((nr,nt, 3)) + np.nan
nt,nx,_ = rer.shape
for t in range(nt):
    rr = r[~np.isnan(rer[t,:,0])]
    if len(rr)>0:
        rmax = rr.max()
        ridx = np.where(r==rmax)[0][0]
        for xx in range(ridx):
            rrer[-xx+ridx-1,t,:] = rer[t,xx,:]

# Display the kymograph
titles = ['CFP', 'YFP', 'RFP']
for c in range(3):
    plt.subplot(3,1,c+1)
    image_artist = plt.imshow(rrer[:,:,c])
    image_artist.set_extent([0,r.max(),0,nt*10])
    plt.colorbar()
    plt.title(titles[c])
    plt.xlabel('Time (frames)')
    plt.ylabel('R (pixels)')
'''

#plt.imshow(nrrer)
plt.show()
np.save('rfluo_var.npy', rfluo_var)
np.save('drfluo_var.npy', drfluo_var)
np.save('fluo0.npy', fluo0)
np.save('fluo1.npy', fluo1)
np.save('dfluo.npy', dfluo)
#np.save('rer.npy', rer)
#np.save('rrer.npy', rer)
#print(f'rstep = {rstep}')

'''
# Compute change in fluorescence
fluo0 = np.zeros((nt, 16, 16, 3)) + np.nan
fluo1 = np.zeros((nt, 16, 16, 3)) + np.nan
for frame in range(nt):
    for ix in range(16):
        for iy in range(16):
            if not np.isnan(vel[ix,iy,frame,0]):
                vx = 0 # round(vel[ix, iy, frame, 0])
                vy = 0 # round(vel[ix, iy, frame, 1])
                f0 = im_all[start_frame+frame*step, ix*16:ix*16+64, iy*16:iy*16+64,[0,2,3]]
                f1 = im_all[start_frame+(frame+1)*step, ix*16+vx:ix*16+64+vx, iy*16+vy:iy*16+64+vy,[0,2,3]]
                #print(fluo0.shape)
                fluo0[frame,ix,iy,:] = f0.mean(axis=(1,2))
                fluo1[frame,ix,iy,:] = f1.mean(axis=(1,2))
dfluo = fluo1 - fluo0

# Construct the growth rate spatial pattern
radpos = np.load('radpos.npy')
growth_rate = np.zeros((nt, 16, 16)) + np.nan
r0 = 31.00122470051893
C = 0 # 0.013742839643935842
mu0 = 2 * vfront / r0 # mu0 = np.load('mu0.npy')
for frame in range(nt):
    for ix in range(16):
        for iy in range(16):
            if not np.isnan(vel[ix,iy,frame,0]):
                r = edt[frame, ix*16:ix*16+64, iy*16:iy*16+64]
                mu = mu0[frame] * np.exp(-r/r0) + C
                mean_mu = mu.mean()
                growth_rate[frame, ix, iy] = mean_mu

# Compute the expression rate
er = np.zeros((nt,16,16,3)) + np.nan
pos = np.load('pos.np.npy')
for c in range(3):
    for frame in range(nt):
        gxfluo,gyfluo = np.gradient(im_all[start_frame+frame*step,:,:,channels[c]])
        for ix in range(16):
            for iy in range(16):
                gx = gxfluo[ix*16:ix*16+64, iy*16:iy*16+64]
                gy = gyfluo[ix*16:ix*16+64, iy*16:iy*16+64]
                vx = vel[ix, iy, frame, 0]
                vy = vel[ix, iy, frame, 1]
                adv =  (gx*vx + gy*vy).mean()
                er[frame,ix,iy,c] = adv # dfluo[frame,ix,iy,c] + adv + (growth_rate[frame,ix,iy] + 0.005) * fluo0[frame,ix,iy,c]

# Save data
np.save('growth_rate.npy', growth_rate)
np.save('er.npy', er)

# Make an animation
from matplotlib.animation import FuncAnimation
fig, axs = plt.subplots(1,4)
er = er[50:,:,:,:]
nt = er.shape[0]
def animate(frame):
    axs[0].imshow(growth_rate[frame,:,:], clim=[np.nanmin(growth_rate), np.nanmax(growth_rate)])
    #axs[0].colorbar()

    axs[1].imshow(er[frame,:,:,0]) #, clim=[np.nanmin(er[:,:,:,0]), np.nanmax(er[:,:,:,0])])
    #axs[1].colorbar()

    axs[2].imshow(er[frame,:,:,1]) #, clim=[np.nanmin(er[:,:,:,1]), np.nanmax(er[:,:,:,1])])
    #axs[2].colorbar()

    axs[3].imshow(er[frame,:,:,2]) #, clim=[np.nanmin(er[:,:,:,2]), np.nanmax(er[:,:,:,2])])
    #axs[3].colorbar()

ani = FuncAnimation(fig, animate, frames=nt, interval=500, repeat=False)
plt.show()
    

'''
