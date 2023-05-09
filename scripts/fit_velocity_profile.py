import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Compute average growth rate of colony at each time point
area = np.load('area.npy')
sarea = savgol_filter(area, 11, 3)
dsarea = savgol_filter(area, 11, 3, deriv=1)
mean_growth_rate = dsarea / area
np.save('mean_growth_rate.npy', mean_growth_rate)

# Compute colony edge velocity
radius = np.load('radius.npy')
vfront = savgol_filter(radius, 5, 3, deriv=1)
np.save('vfront.npy', vfront)

# Normalize velocity by edge vel
vmag = np.load('vmag.npy')
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

edt = np.load('edt.npy')
res = least_squares(residual_func(edt, nvmag, nt), x0=(np.log(50),0))
r0 = np.exp(res.x[0])
C = res.x[1]

print(f'r0 = {r0}, C = {C}')

# Make a plot to see how good the fit is
radpos = np.load('radpos.npy')
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
plt.show()



