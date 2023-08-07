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

start_frame = 50
offset = 0
step = 1

vfront = np.load('vfront.npy')
vfront = vfront[offset:]
print(vfront.shape)

# Normalize velocity by edge vel
vmag = np.load('vmag.npy')
vmag = vmag[offset:,:,:]
nt,nx,ny = vmag.shape

svmag = np.zeros_like(vmag) + np.nan
for t in range(nt):
    for ix in range(1,nx-1):
        for iy in range(1,ny-1):
            svmag[t,ix,iy] = np.nanmean(vmag[t,ix-1:ix+2,iy-1:iy+2])

nvmag = np.zeros_like(svmag)
for frame in range(nt):
    nvmag[frame,:,:] = svmag[frame,:,:] / vfront[frame*step + start_frame]

radpos = np.load('radpos.npy')
radpos = radpos[offset:,:,:]
#nvmag[~np.isnan(radpos)] = np.nan
#svmag[~np.isnan(radpos)] = np.nan


rmax = np.load('radius.npy')
rmax = rmax[offset:]

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
                        r = edt[frame, ix*24:ix*24+48, iy*24:iy*24+48]
                        B = 1 / (1 - np.exp(-rmax[t]/r0))
                        model_vmag = 1 + B * (np.exp(-r/r0) - 1)
                        mean_model_vmag = np.nanmean(model_vmag)
                        res.append(mean_model_vmag - nvmag[frame, ix, iy])
        return res
    return residuals



edt = np.load('edt.npy')
edt = edt[offset:,:,:]
res = least_squares(residual_func(edt, nvmag, nt, nx, ny), x0=(np.log(50),))
r0 = np.exp(res.x[0])
C = 0 #res.x[1]

print(f'r0 = {r0}, C = {C}')


# Make a plot to see how good the fit is
colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
i = 0
times = [0] #[0,9,19,29]
for t in times:
    plt.subplot(2,2,i+1)
    x_model = np.zeros((0,))    
    y_model = np.zeros((0,))
    x_data = np.zeros((0,))
    y_data = np.zeros((0,))
    
    for ix in range(nx):
        for iy in range(ny):
            if not np.isnan(svmag[t,ix,iy]):
                r = edt[t, ix*24:ix*24+48, iy*24:iy*24+48]
                x_data = np.append(x_data, np.nanmean(r))
                y_data = np.append(y_data, svmag[t,ix,iy])
    plt.plot(x_data, y_data, '.', alpha=0.2) #, color=colours[i])

    for r in np.linspace(0, rmax[t], 100):
        B = 1 / (1 - np.exp(-rmax[t]/r0))
        model_vmag = 1 + B * (np.exp(-r/r0) - 1)
        mean_model_vmag = np.nanmean(model_vmag)
        x_model = np.append(x_model, np.nanmean(r))
        y_model = np.append(y_model, vfront[start_frame + step*t] * mean_model_vmag)
    plt.plot(x_model, y_model, 'k--') #, color=colours[i])

    i += 1

    plt.title(f't = {t}')
    plt.xlabel('Radial position')
    plt.ylabel('$v/v_{front}$')
plt.tight_layout()
plt.show()

y_model = np.zeros((0,))
y_data = np.zeros((0,))
for t in range(nt):
    for ix in range(nx):
        for iy in range(ny):
            if not np.isnan(nvmag[t,ix,iy]):
                r = edt[t, ix*24:ix*24+48, iy*24:iy*24+48]
                B = 1 / (1 - np.exp(-rmax[t]/r0))
                model_vmag = 1 + B * (np.exp(-r/r0) - 1)
                mean_model_vmag = np.nanmean(model_vmag)
                y_model = np.append(y_model, vfront[start_frame + step*t] * mean_model_vmag)
                y_data = np.append(y_data, svmag[t,ix,iy])

plt.plot(y_model, y_data, '.', alpha=0.1)
plt.plot([0,y_model.max()], [0,y_model.max()], 'k--')
plt.xlabel('Model $v$')
plt.ylabel('Velocimetry $v$')
#plt.xscale('log')
#plt.yscale('log')
plt.show()

mu0 = np.zeros((nt,))
for t in range(nt):
    B = 1 / (1 - np.exp(-rmax[t]/r0))
    mu0[t] = 2 * vfront[start_frame + step*t] / r0 * B

# The edge growth rate = 2 * edge velocity / r0 
np.save('mu0.npy', mu0)



