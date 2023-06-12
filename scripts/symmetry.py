import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

start_frame = 100
step = 1

# Load the velocity field
#vel = np.load('vel.np.npy')
#_,_,nt,_ = vel.shape

# Load the radial velocity magnitude
#vmag = np.load('vmag.npy')

nt = 1

# Load the tif image from the microscope
fname = '10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.tif' # 'Fused_12_13_14_15_1024.bgcorr.tiff'
im_all = imread('../' + fname)
print(im_all.shape)
channels = [0,2,3]

'''
radpos = np.load('radpos.npy')

# Compute fluorescence of each grid square
fluo0 = np.zeros((nt, 19, 32, 3)) + np.nan
fluo1 = np.zeros((nt, 19, 32, 3)) + np.nan
for frame in range(nt):
    for ix in range(19):
        for iy in range(32):
            #if not np.isnan(vmag[frame,ix,iy]):
            if not np.isnan(radpos[frame, ix, iy]):
                f0 = im_all[start_frame+frame*step, ix*32:ix*32+64, iy*32:iy*32+64,[0,1,2]]
                f1 = im_all[start_frame+(frame+1)*step, ix*32:ix*32+64, iy*32:iy*32+64,[0,1,2]]
                fluo0[frame,ix,iy,:] = f0.mean(axis=(1,2))
                fluo1[frame,ix,iy,:] = f1.mean(axis=(1,2))
dfluo = fluo1 - fluo0

np.save('fluo0.npy', fluo0)
np.save('dfluo.npy', dfluo)

# Compute radial average of fluorescence and derivative, df/ft
rmax = np.nanmax(radpos)
print(rmax)
nr = 32
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

np.save('rfluo.npy', rfluo)
np.save('drfluo.npy', drfluo)
np.save('rfluo_var.npy', rfluo_var)
np.save('drfluo_var.npy', drfluo_var)

# Compute angular average of fluorescence and derivative, df/ft
pos = np.load('pos.np.npy')
print(pos.shape)
na = 16
astep = 2 * np.pi / na
afluo = np.zeros((nt,na, 3))
dafluo = np.zeros((nt,na, 3))
afluo_var = np.zeros((nt,na, 3))
dafluo_var = np.zeros((nt,na, 3))
for frame in range(nt):
    ang = np.arctan2(pos[:,:,frame,1] - 500, pos[:,:,frame,0] - 320)
    ang[ang<0] += 2 * np.pi
    for c in range(3):
        f = fluo0[frame,:,:,c]
        df = dfluo[frame,:,:,c]
        for a in range(na):
            ang0 = a * astep
            ang1 = (a + 1) * astep
            print(ang0, ang1)
            print(np.sum((ang>=ang0)*(ang<ang1)))
            afluo[frame,a,c] = np.nanmean(f[(ang>=ang0)*(ang<ang1)])
            dafluo[frame,a,c] = np.nanmean(df[(ang>=ang0)*(ang<ang1)])
            afluo_var[frame,a,c] = np.nanvar(f[(ang>=ang0)*(ang<ang1)])
            dafluo_var[frame,a,c] = np.nanvar(df[(ang>=ang0)*(ang<ang1)])
'''
#x,y = np.meshgrid(np.arange(1024), np.arange(632))
x,y = np.meshgrid(np.arange(1024), np.arange(1024))

#cx,cy = 320,500
#ang = np.arctan2(x-cy, y-cx)
#ang[ang<0] += 2 * np.pi
#rad = np.sqrt((x-cy)**2 + (y-cx)**2)
#rmax = 300

na = 16
nr = 32
astep = 2 * np.pi / na

mask = imread('../C2-10x_1.5x_-5_pAAA_MG_1_MMStack_Pos9.ome.mask.tif')
mask = mask>0

# Compute transformation to polar coordinates
rafluo = np.zeros((nt,na,nr,3))
rafluo_var = np.zeros((nt,na,nr,3))

for frame in range(nt):
    edt = distance_transform_edt(mask[start_frame + frame*step,:,:])
    rmax = np.nanmax(edt) - 20
    rstep = rmax / nr

    cx = np.nanmean(x[mask[start_frame+frame*step,:,:]])
    cy = np.nanmean(y[mask[start_frame+frame*step,:,:]])
    ang = np.arctan2(x-cx, y-cy)
    ang[ang<0] += 2 * np.pi
    rad = np.sqrt((x-cy)**2 + (y-cx)**2)
 
    for c in range(3):
        f =  im_all[start_frame+frame*step,:,:,channels[c]]
        plt.imshow(f)
        plt.show()
        for a in range(na):
            for r in range(nr):
                ang0 = a * astep
                ang1 = (a + 1) * astep
                rad0 = r * rstep
                rad1 = (r + 1) * rstep
                idx = (ang>=ang0)*(ang<ang1)*(rad>=rad0)*(rad<rad1)
                rafluo[frame,a,r,c] = np.nanmean(f[idx])
                rafluo_var[frame,a,r,c] = np.nanvar(f[idx])

np.save(f'{fname}_rafluo.npy', rafluo)
np.save(f'{fname}_afluo_var.npy', rafluo_var)
