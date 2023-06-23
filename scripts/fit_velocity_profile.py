import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Compute average growth rate of colony at each time point
#area = np.load('area.npy')
#sarea = savgol_filter(area, 5, 3)
#dsarea = savgol_filter(area, 5, 3, deriv=1)
#mean_growth_rate = dsarea / area
#np.save('mean_growth_rate.npy', mean_growth_rate)

# Compute colony edge velocity
#radius = np.load('radius.npy')
#vfront = savgol_filter(radius, 5, 3, deriv=1)
#np.save('vfront.npy', vfront)
#vfront = 6.585722451885091 
#vfront = 65.8267122272761 / 60 * 10

start_frame = 40
step = 1

vfront = np.load('vfront.npy')
print(vfront.shape)

# Normalize velocity by edge vel
vmag = np.load('vmag.npy')
nt,nx,ny = vmag.shape

svmag = np.zeros_like(vmag)
for t in range(nt):
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            svmag[t,ix,iy] = np.nanmean(vmag[t,ix-1:ix+2,iy-1:iy+2])

nvmag = np.zeros_like(svmag)
for frame in range(nt):
    nvmag[frame,:,:] = svmag[frame,:,:] / vfront[frame*step + start_frame]

radpos = np.load('radpos.npy')
nvmag[radpos==0] = np.nan

rmax = np.load('radius.npy')

# Fit an exponential decay model to the velocity data
def residual_func(edt, nvmag, nt, nx, ny):
    def residuals(x):
        r0 = np.exp(x[0])
        C = 0 #x[1]
        res = []
        for frame in range(nt):
            for ix in range(nx):
                for iy in range(ny):
                    if not np.isnan(nvmag[frame,ix,iy]):
                        r = edt[frame, ix*32:ix*32+64, iy*32:iy*32+64]
                        B = 1 / (1 - np.exp(-rmax[t]/r0))
                        model_vmag = 1 + B * (np.exp(-r/r0) - 1)
                        mean_model_vmag = model_vmag.mean()
                        res.append(mean_model_vmag - nvmag[frame, ix, iy])
        return res
    return residuals



edt = np.load('edt.npy')
res = least_squares(residual_func(edt, nvmag, nt, nx, ny), x0=(np.log(50),))
r0 = np.exp(res.x[0])
C = 0 #res.x[1]

print(f'r0 = {r0}, C = {C}')


# Make a plot to see how good the fit is
x = radpos[~np.isnan(nvmag)].ravel()
y = np.zeros((0,))
yn = np.zeros((0,))
for t in range(nt):
    for ix in range(nx):
        for iy in range(ny):
            if not np.isnan(nvmag[t,ix,iy]):
                r = edt[t, ix*32:ix*32+64, iy*32:iy*32+64]
                B = 1 / (1 - np.exp(-rmax[t]/r0))
                model_vmag = 1 + B * (np.exp(-r/r0) - 1)
                mean_model_vmag = model_vmag.mean()
                yn = np.append(yn, mean_model_vmag)
                y = np.append(y, vfront[start_frame + step*t] * mean_model_vmag)

plt.plot(x, nvmag[~np.isnan(nvmag)], '.', alpha=0.2)
plt.plot(x, yn, '.')
plt.xlabel('Radial position')
plt.ylabel('$v/v_{front}$')
plt.show()

plt.plot(y, svmag[~np.isnan(nvmag)], '.', alpha=0.2)
plt.plot([0,y.max()], [0,y.max()], 'k--')
plt.xlabel('Model $v$')
plt.ylabel('Velocimetry $v$')
plt.xscale('log')
plt.yscale('log')
plt.show()

mu0 = np.zeros((nt,))
for t in range(nt):
    B = 1 / (1 - np.exp(-rmax[t]/r0))
    mu0[t] = 2 * vfront[start_frame + step*t] / r0 * B

# The edge growth rate = 2 * edge velocity / r0 
np.save('mu0.npy', mu0)



