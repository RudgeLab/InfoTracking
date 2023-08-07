import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

'''
#cim = imread('/Users/timrudge/Pictures/10x_1.5x_pLPT20&41_DHL_TiTweez_1_MMStack_Pos8.centre.ome.tif')
#cim = imread('/Volumes/General/Microscopy/10x_1.5x_pLPT20&41_DHL_TiTweez_1_MMStack_Pos8.centre.ome.tif')
#cim = imread('/Volumes/General/Microscopy/Rep_44ch_100cc_10X_1.5X_1_MMStack_Pos14.ome.crop.centre.tif')
#cim = imread('/Volumes/General/Microscopy/10x_1.0x_-.5div2_pLPT20_MC_TiTweez_1_MMStack_Pos7.centre.ome.tif')
#cim = imread('/Volumes/General/Microscopy/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.centre.ome.tif')
cim = imread('/Volumes/General/Microscopy/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.centre.ome.tif')
#bgim = imread('/Volumes/General/Microscopy/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.tif')
#bg = bgim[0,:100,:100,:].mean(axis=(0,1))
cim = cim.mean(axis=(1,2))
#for c in range(3):
#    cim[:,c] = cim[:,c] - bg[c]
cim = cim[30:100,:]
cfp = cim[:,[3]] # - cim[:,[2]].min()
yfp = cim[:,[2]] # - cim[:,[1]].min()
'''

#edge_prof = np.load('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.edge.sprofile.npy')
kymo = np.load('kymo.npy')
#edge_prof = savgol_filter(kymo[20,:,:], 21, 3)
#dedge_prof = savgol_filter(kymo[20,:,:], 21, 3, deriv=1)
#dedge_prof = edge_prof # np.load('/Volumes/General/Analysis/10x_1x_pLPT20&41_MC_TiTweez_1_MMStack_Pos0.edge.dsprofile.npy')
yfp = savgol_filter(kymo[20,:,0], 21, 3) # edge_prof[:,[0]]
cfp = savgol_filter(kymo[20,:,1], 21, 3) # edge_prof[:,[1]]
dyfp = savgol_filter(kymo[20,:,0], 21, 3, deriv=1) # dedge_prof[:,[0]]
dcfp = savgol_filter(kymo[20,:,1], 21, 3, deriv=1) # dedge_prof[:,[1]]

gamma = np.log(2) / (24*60/10)
prof = yfp # (dyfp + yfp*gamma) / (dcfp + cfp*gamma)
refprof = dcfp + gamma * cfp
nt = prof.shape[0]
dt = 24 / nt
print(nt, dt)

#for c in range(nc):
#    prof[:,c] = prof[:,c] - prof[:,c].min()

plt.subplot(1,2,1)
plt.plot(prof)
plt.subplot(1,2,2)
plt.plot(refprof)
plt.show()

# Promoters
# Plac -> CFP, TetR
# Ptet -> YFP, cI
# PR -> RFP, LacI

# Channels:
# 0 = CFP
# 1 = YFP
# 2 = RFP

# Interactions
# 0 -| 1 -| 2 -| 0

def residuals(prof, refprof, tt, eps_model, eps_alpha):
    def func(x):
        data = prof
        #gamma = 0.5 # np.exp(x[0])
        alpha = np.exp(x[0:3])
        beta = np.exp(x[3:6])
        K =     np.exp(x[6:9])
        p0 = np.exp(x[9:12])
        p0 = np.append(p0, p0)
        #t0 = np.exp(x[12])
        deg_rate = np.exp(x[12])
        #mu0 = np.exp(x[13]) # np.log(2)/(28/10)
        #T0 = np.exp(x[15]) 
        n = 4 #np.exp(x[13])
        print(alpha, beta, K, p0, deg_rate)

        rp = interp1d(tt, refprof[:], bounds_error=False, fill_value='extrapolate')

        def deriv(y, t):
            r = y[:3]
            drdt = np.zeros_like(r)
            f = y[3:]
            dfdt = np.zeros_like(f)
            for idx1 in range(3):
                idx2 = (idx1 + 1) % 3
                gamma_f =  np.log(2) / (24*60/10)
                gamma_r = deg_rate #[idx2] # mu0 * np.exp(-t/t0) + deg_rate # mu0 / ( 1 + np.exp( (t-t0)*mu0/2 ) ) + deg_rate
                drdt[idx2] = rp(t) * (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_r * r[idx2]
                dfdt[idx2] = rp(t) * (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_f * f[idx2]
            dydt = np.zeros_like(y)
            dydt[:3] = drdt
            dydt[3:] = dfdt
            return dydt
        #tt = np.append(np.linspace(-8, 0, 100), t)
        #tt = np.append(tt, np.linspace(tt.max()+0.1, tt.max()*4, 100))
        model = odeint(deriv, p0, tt)
        residual = (data[:] - model[:,3]).ravel()
        residual = np.append(residual, eps_model * model.ravel())
        #residual = np.append(residual, eps_alpha * alpha / K)
        return residual
    return func

gamma0 = [np.log(2) / (12 * 60 / 10)] * 3
alpha0 = [1e2] * 3
beta0 = [1e-6] * 3
K0 = [1]*3
p0 = [2000, 2000, 2000]
t0 = [20]
deg_rate0 = [0.25]
mu0 = [0.1] # [np.log(2)/(28/10)]
T0 = [10]
n0 = [4]

x0 = np.array(alpha0 + beta0 + K0 + p0 + deg_rate0)
x0 = np.log(x0)

#gamma = np.zeros((3,))
tt = np.arange(nt)
res = least_squares(residuals(prof, refprof, tt, eps_model=1e-2, eps_alpha=0), x0=x0) #, diff_step=[1e-1]*len(x0))
#print(res)
    
alpha = np.exp(res.x[:3])
beta = np.exp(res.x[3:6])
K = np.exp(res.x[6:9])
p0 = np.exp(res.x[9:12])
p0 = np.append(p0, p0)
#t0 = np.exp(res.x[12])
deg_rate = np.exp(res.x[12])
#mu0 = np.exp(res.x[13])
#T0 = np.exp(res.x[15])
n = 4 # np.exp(res.x[13])
print(alpha, beta, p0, K, deg_rate, alpha/K)

rp = interp1d(tt, refprof[:], bounds_error=False, fill_value='extrapolate')

def deriv(y, t):
    r = y[:3]
    drdt = np.zeros_like(r)
    f = y[3:]
    dfdt = np.zeros_like(f)
    for idx1 in range(3):
        idx2 = (idx1 + 1) % 3
        gamma_f =  np.log(2) / (24*60/10)
        gamma_r = deg_rate #[idx2] # mu0 * np.exp(-t/t0) + deg_rate # mu0 / ( 1 + np.exp( (t-t0)*mu0/2 ) ) + deg_rate
        drdt[idx2] = rp(t) * (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_r * r[idx2]
        dfdt[idx2] = rp(t) * (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_f * f[idx2]
    dydt = np.zeros_like(y)
    dydt[:3] = drdt
    dydt[3:] = dfdt
    return dydt
tt2 = tt # np.linspace(0,24,nt)
#tt2 = np.append(np.linspace(-8, 0, 100), tt2)
#tt2 = np.append(tt2, np.linspace(tt.max()+0.1, tt.max()*8, 100))

model = odeint(deriv, p0, tt2)
data = prof

residual = (data[:] - model[:nt,3]).ravel()
print(np.sum(np.sqrt(residual*residual)))

plt.plot(residual)
plt.show()

plt.subplot(1,2,1)
colors = ['b', 'g', 'r']
#for c in range(nc):
plt.plot(tt, data[:]/K[0], color=colors[0])
for c in range(3):
    plt.plot(tt2, model[:,c+3]/K[c], '--', color=colors[c])

plt.subplot(1,2,2)
#for c in range(nc):
plt.plot(tt, data[:]/K[0], color=colors[0])
for c in range(3):
    plt.plot(tt, model[:-100,c+3]/K[c], '--', color=colors[c])

#plt.yscale('log')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.plot(model[:,0], model[:,1], model[:,2], '--')
#plt.plot(data[:,0], data[:,1], data[:,2], '-')
plt.show()

'''
# Construct L-curve
nl = 10
eps = np.logspace(-6, 0, nl)
norms = np.zeros((0,))
errs = np.zeros((0,))

for e in eps:
    res = least_squares(residuals(prof, tt, eps=e), x0=x0)
    alpha = np.exp(res.x[:3])
    beta = np.exp(res.x[3:6])
    K = np.exp(res.x[6:9])
    p0 = np.exp(res.x[9:12])
    p0 = np.append(p0, p0)
    deg_rate = np.exp(res.x[12:14])
    print(alpha, beta, p0, K, deg_rate)

    err = np.sum(np.sqrt(res.fun * res.fun))
    model = odeint(deriv, p0, tt2)
    norm = np.sum(np.sqrt((model * model).ravel()))
    errs = np.append(errs, err)
    norms = np.append(norms, norm)

print(eps)
plt.plot(norms, errs, '+')
plt.xscale('log')
plt.yscale('log')
plt.show()
'''
