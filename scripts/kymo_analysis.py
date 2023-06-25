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
nt = 120

im_all = imread('../Microscopy/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.tif')
#im_all = imread('../Microscopy/10x_1.0x_pLPT20_DHL_TiMain_1_MMStack_Pos5.ome.tif')
mask_all = imread('10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.contour.mask.tif')
#mask_all = imread('10x_1.0x_pLPT20_DHL_TiMain_1_MMStack_Pos5.ome.contour.mask.tif')

x,y = np.meshgrid(np.arange(1200), np.arange(1200))
nr = 32
kymo = np.zeros((nr,nt,2))
rkymo = np.zeros((nr,nt))
nrkymo = np.zeros((nr,nt))
rsteps = np.zeros((nt,))
for t in range(nt):
    print(f'Processing frame {t}')
    im = im_all[start_frame + t,:,:,1:3].astype(float)
    mask = mask_all[start_frame + t,:,:].astype(float)

    for c in range(2):
        cim = im[:,:,c]
        cim = median_filter(cim, size=5)
        bg = cim[:100,:100].mean(axis=(0,1))
        cim = cim - bg
        cim[mask==0] = np.nan
        im[:,:,c] = cim

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

    #edt = distance_transform_edt(mask)
    rmin = 0
    rmax = 600 #edt.max()
    rstep = (rmax - rmin) / nr
    rsteps[t] = rstep

    rr = np.zeros((nr,))
    for r in range(nr):
        im0 = im[:,:,0]
        rim0 = im0[(R>=(rmin + r*rstep))*(R<(rmin + (r+1)*rstep))]
        kymo[r,t,0] = np.mean(rim0)
        im1 = im[:,:,1]
        rim1 = im1[(R>=(rmin + r*rstep))*(R<(rmin + (r+1)*rstep))]
        kymo[r,t,1] = np.mean(rim1)
        rr[r] = kymo[r,t,0] / kymo[r,t,1]
    rkymo[:,t] = rr
    nrkymo[:,t] = (rr - np.nanmin(rr)) / (np.nanmax(rr) - np.nanmin(rr))
    r = np.linspace(0, rmax, nr)
    r = r[~np.isnan(rr)]
    rr = rr[~np.isnan(rr)]
    #plt.subplot(1,4,4)
    #plt.plot(rr)
    #plt.tight_layout()
    #plt.show()


    #corr = correlate(rr, rr)
    #plt.plot(corr)
    #plt.show()

    #frr = fft(rr - rr.mean(), n=256)
    #ps = frr * frr.conjugate()
    #ps = ps.real
    #plt.plot(ps)
    #plt.show()

    #def func(x):
    #    wavelength = x[0]
    #    phase = x[1]
    #    amp = x[2]
    #    mean = x[3]
    #    model = mean + amp * np.cos(r * 2 * np.pi / wavelength + phase)
    #    return model - rr

    #res = least_squares(func, x0=[rmax, 0, 0.5, 0.25])
    #wavelength = res.x[0]
    #phase = res.x[1]
    #amp = res.x[2]
    #mean = res.x[3]

    #print(f'wavelength = {wavelength}, phase = {phase}, amp = {amp}, mean = {mean}')

    #plt.plot(r, rr)
    #plt.plot(r, mean + amp * np.cos(r * 2 * np.pi / wavelength + phase), 'k--')
    #plt.show()

np.save('kymo.npy', kymo)
np.save('rkymo.npy', rkymo)
np.save('nrkymo.npy', nrkymo)

x = nrkymo[:,-1]
phases = np.zeros((nt,)) + np.nan
for t in range(nt):
    y = nrkymo[:,t]
    xidx = ~np.isnan(x)
    yidx = ~np.isnan(y)
    nx = np.sum(xidx)
    ny = np.sum(yidx)
    if nx>10 and ny>10:
        corr = correlate(x[xidx], y[yidx])  / x[xidx].std() / y[yidx].std() / nx / ny
        lags = correlation_lags(nx, ny)
        idx = np.argmax(corr)
        phase = lags[idx]
        phases[t] = phase
        plt.plot(lags, corr)
plt.show()

pks = np.zeros((nt,)) + np.nan
for t in range(40,nt):
    f = nrkymo[:,t]
    idx = ~np.isnan(f)
    n = np.sum(idx)
    if n>2:
        pks[t] = np.argmax(f[idx])

edge = np.zeros((nt,)) + np.nan
for t in range(nt):
    x = nrkymo[:,t]
    valid = ~np.isnan(x)
    idx = np.where(valid)[0]
    if len(idx)>2:
        idx_range = idx[-2:]
        edge[t] = x[idx_range].mean()

centre = nrkymo[:2,:].mean(axis=0)
plt.plot(edge[::2])
plt.plot(centre[::2])
plt.legend(['Edge', 'Centre'])
plt.show()

ex = edge[:]
ex = ex[~np.isnan(ex)]
ex = (ex - ex.mean()) / ex.std()
fedge = fft(ex, n=len(ex)*16)
ps_edge = fedge * fedge.conjugate()

cx = centre[:]
cx = cx[~np.isnan(cx)]
cx = (cx - cx.mean()) / cx.std()
fcentre = fft(cx, n=len(cx)*16)
ps_centre = fcentre * fcentre.conjugate()

cfreq = fftfreq(len(cx)*16) / 10 * 60
efreq = fftfreq(len(ex)*16) / 10 * 60

f_edge = np.abs(efreq[np.argmax(ps_edge.real)])
T_edge = 1 / f_edge
f_centre = np.abs(cfreq[np.argmax(ps_centre.real)])
T_centre = 1 / f_centre

print(f'T_edge = {T_edge}, T_centre = {T_centre}')

plt.plot(efreq[(efreq>=0)*(efreq<0.25)], ps_edge[(efreq>=0)*(efreq<0.25)])
plt.plot(cfreq[(cfreq>=0)*(cfreq<0.25)], ps_centre[(cfreq>=0)*(cfreq<0.25)])
plt.plot(f_edge, ps_edge.real[np.argmax(ps_edge.real)], 'x')
plt.plot(f_centre, ps_centre.real[np.argmax(ps_centre.real)], '+')
plt.legend(['Edge', 'Centre'])
plt.show()

filt_fedge = np.zeros_like(fedge)
eidx = np.argmax(ps_edge.real)
filt_fedge[eidx] = fedge[eidx]
ne = len(ex)*16
ifilt_fedge = ifft(filt_fedge) * np.sqrt(ne)

filt_fcentre = np.zeros_like(fcentre)
cidx = np.argmax(ps_centre.real)
filt_fcentre[cidx] = fcentre[cidx]
nc = len(cx)*16
ifilt_fcentre = ifft(filt_fcentre) * np.sqrt(nc)

plt.plot(ex[::2])
plt.plot(cx[::2])
plt.plot(ifilt_fedge.real[:144:2])
plt.plot(ifilt_fcentre.real[:144:2])


plt.legend(['Edge', 'Centre'])
plt.show()


#plt.figure(figsize=(12,5))
plt.subplot(2,2,1)
plt.imshow(kymo[:,:,0])
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(kymo[:,:,1])
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(rkymo[:,:])
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(nrkymo[:,:])
plt.colorbar()

plt.savefig('kymos.png', dpi=300)
plt.show()
