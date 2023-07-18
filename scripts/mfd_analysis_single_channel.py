import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#cim = imread('/Users/timrudge/Pictures/10x_1.5x_pLPT20&41_DHL_TiTweez_1_MMStack_Pos8.centre.ome.tif')
#cim = imread('/Volumes/General/Microscopy/10x_1.5x_pLPT20&41_DHL_TiTweez_1_MMStack_Pos8.centre.ome.tif')
#cim = imread('/Volumes/General/Microscopy/Rep_44ch_100cc_10X_1.5X_1_MMStack_Pos14.ome.crop.centre.tif')
cim = imread('/Volumes/General/Microscopy/10x_1.0x_-.5div2_pLPT20_MC_TiTweez_1_MMStack_Pos7.centre.ome.tif')
#cim = imread('/Volumes/General/Microscopy/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.centre.ome.tif')
#bgim = imread('/Volumes/General/Microscopy/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.tif')
#bg = bgim[0,:100,:100,:].mean(axis=(0,1))
cim = cim.mean(axis=(1,2))
#for c in range(3):
#    cim[:,c] = cim[:,c] - bg[c]
cim = cim[30:,:]
cfp = cim[:,[2]] # - cim[:,[2]].min()
yfp = cim[:,[1]] # - cim[:,[1]].min()

prof = yfp # cfp
nt,nc = prof.shape
dt = 24 / nt
print(nt, nc, dt)

sprof = np.zeros_like(prof)
dsprof = np.zeros_like(prof)
for c in range(nc):
    x = prof[:,c]
    sprof[:,c] = savgol_filter(x, 11, 3)
    dsprof[:,c] = savgol_filter(x, 11, 3, deriv=1)
    sprof[:,c] = sprof[:,c] - sprof[:,c].min()
    prof[:,c] = prof[:,c] - prof[:,c].min()

plt.subplot(1,3,1)
plt.plot(sprof)
plt.subplot(1,3,2)
plt.plot(dsprof)
plt.subplot(1,3,3)
plt.legend(['CFP', 'YFP', 'RFP'])
gamma = 0.5
plt.plot(dsprof + gamma * sprof)
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


def residuals(sprof, t, eps):
    def func(x):
        data = sprof
        #gamma = 0.5 # np.exp(x[0])
        alpha = np.exp(x[0:3])
        beta = np.exp(x[3:6])
        K = np.exp(x[6:9])
        p0 = np.exp(x[9:12])
        #p0 = np.append(p0, p0)
        #t0 = np.exp(x[12])
        deg_rate = np.exp(x[12])
        #mu0 = np.exp(x[13]) # np.log(2)/(28/10)
        #T0 = np.exp(x[15]) 
        n = 4 #np.exp(x[13])
        #print(alpha, beta, K, p0, deg_rate)
        def deriv(y, t):
            r = y[:3]
            drdt = np.zeros_like(r)
            #f = y[3:]
            #dfdt = np.zeros_like(f)
            for idx1 in range(3):
                idx2 = (idx1 + 1) % 3
                gamma_r = deg_rate #[idx2] # mu0 * np.exp(-t/t0) + deg_rate # mu0 / ( 1 + np.exp( (t-t0)*mu0/2 ) ) + deg_rate
                gamma_f = gamma_r
                drdt[idx2] = (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_r * r[idx2]
                #dfdt[idx2] = (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_f * f[idx2]
            #dydt = np.zeros_like(y)
            #dydt[:3] = drdt
            #dydt[3:] = dfdt
            return drdt
        #tt = np.append(np.linspace(-8, 0, 100), t)
        #tt = np.append(tt, np.linspace(tt.max()+0.1, tt.max()*4, 100))
        model = odeint(deriv, p0, tt)
        residual = (data[:,0] - model[:,0]).ravel()
        residual = np.append(residual, eps * model.ravel())
        return residual
    return func

gamma0 = [np.log(2) / (12 * 60 / 10)] * 3
alpha0 = [1e4] * 3
beta0 = [1e-6] * 3
K0 = [1e2]*3
p0 = [2000, 2000, 2000]
t0 = [20]
deg_rate0 = [0.25]
mu0 = [0.1] # [np.log(2)/(28/10)]
T0 = [10]
n0 = [4]

x0 = np.array(alpha0 + beta0 + K0 + p0 + deg_rate0)
x0 = np.log(x0)

#gamma = np.zeros((3,))
tt = np.linspace(0, 24, nt) # np.arange(nt)
res = least_squares(residuals(prof, tt, eps=1e-2), x0=x0) #, diff_step=[1e-3]*len(x0))
#print(res)

alpha = np.exp(res.x[:3])
beta = np.exp(res.x[3:6])
K = np.exp(res.x[6:9])
p0 = np.exp(res.x[9:12])
#p0 = np.append(p0, p0)
#t0 = np.exp(res.x[12])
deg_rate = np.exp(res.x[12])
#mu0 = np.exp(res.x[13])
#T0 = np.exp(res.x[15])
n = 4 # np.exp(res.x[13])
print(alpha, beta, p0, K, deg_rate)

def deriv(y, t):
    r = y[:3]
    drdt = np.zeros_like(r)
    #f = y[3:]
    #dfdt = np.zeros_like(f)
    for idx1 in range(3):
        idx2 = (idx1 + 1) % 3
        gamma_r = deg_rate #[idx2] # mu0 * np.exp(-t/t0) + deg_rate # mu0 / ( 1 + np.exp( (t-t0)*mu0/2 ) ) + deg_rate
        gamma_f = gamma_r
        drdt[idx2] = (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_r * r[idx2]
        #dfdt[idx2] = (alpha[idx1] + beta[idx1] * ((r[idx1] / K[idx1])**n)) / (1 + (r[idx1] / K[idx1])**n) - gamma_f * f[idx2]
    #dydt = np.zeros_like(y)
    #dydt[:3] = drdt
    #dydt[3:] = dfdt
    return drdt
tt2 = np.linspace(0,24,nt)
#tt2 = np.append(np.linspace(-8, 0, 100), tt2)
tt2 = np.append(tt2, np.linspace(tt.max()+0.1, tt.max()*8, 100))

model = odeint(deriv, p0, tt2)
data = prof

residual = (data[:,0] - model[:nt,0]).ravel()
print(np.sum(np.sqrt(residual*residual)))

plt.plot(residual)
plt.show()

plt.subplot(1,2,1)
colors = ['b', 'g', 'r']
for c in range(nc):
    plt.plot(tt, data[:,c], color=colors[c])
for idx1 in range(3):
    idx2 = (idx1 + 1) % 3
    plt.plot(tt2, model[:,idx2], '--', color=colors[idx2])

plt.subplot(1,2,2)
for c in range(nc):
    plt.plot(tt, data[:,c], color=colors[c])
for idx1 in range(3):
    idx2 = (idx1 + 1) % 3
    plt.plot(tt, model[:-100,idx2], '--', color=colors[idx2])

#plt.yscale('log')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.plot(model[:,0], model[:,1], model[:,2], '--')
#plt.plot(data[:,0], data[:,1], data[:,2], '-')
plt.show()

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
