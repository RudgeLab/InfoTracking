import numpy as np
from skimage.io import imread, imsave
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#cim = imread('/Users/timrudge/Pictures/10x_1.5x_pLPT20&41_DHL_TiTweez_1_MMStack_Pos8.centre.ome.tif')
#cim = imread('/Volumes/General/Microscopy/10x_1.5x_pLPT20&41_DHL_TiTweez_1_MMStack_Pos8.centre.ome.tif')
cim = imread('/Volumes/General/Microscopy/Rep_44ch_100cc_10X_1.5X_1_MMStack_Pos14.ome.crop.centre.tif')
cim = cim.mean(axis=(1,2))
prof = cim[:35,[0,2,3]]

nt,nc = prof.shape

sprof = np.zeros_like(prof)
dsprof = np.zeros_like(prof)
for c in range(3):
    x = prof[:,c]
    sprof[:,c] = savgol_filter(x, 11, 3)
    dsprof[:,c] = savgol_filter(x, 11, 3, deriv=1)
    sprof[:,c] = sprof[:,c] - sprof[:,c].min()

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


def residuals(sprof, t):
    def func(x):
        data = sprof
        #gamma = 0.5 # np.exp(x[0])
        alpha = np.exp(x[0:3])
        beta = np.exp(x[3:6])
        K = np.exp(x[6:9])
        p0 = np.exp(x[9:12])
        #t0 = np.exp(x[12])
        deg_rate = np.exp(x[12:15])
        #mu0 = np.exp(x[13]) # np.log(2)/(28/10)
        #T0 = np.exp(x[15]) 
        n = 4 #np.exp(x[13])
        #print(alpha, beta, K, p0, deg_rate)
        def deriv(y, t):
            dydt = np.zeros_like(y)
            for idx1 in range(3):
                idx2 = (idx1 + 1) % 3
                gamma = deg_rate[idx2] # mu0 * np.exp(-t/t0) + deg_rate # mu0 / ( 1 + np.exp( (t-t0)*mu0/2 ) ) + deg_rate
                dydt[idx2] = (alpha[idx1] + beta[idx1] * ((y[idx1] / K[idx1])**n)) / (1 + (y[idx1] / K[idx1])**n) - gamma * y[idx2]
            return dydt
        #model = p0[idx1] + np.cumsum(model)
        model = odeint(deriv, p0, t)
        return (data - model).ravel()
    return func

gamma0 = [np.log(2) / (12 * 60 / 10)] * 3
alpha0 = [1e2] * 3
beta0 = [1e-1] * 3
K0 = [1] * 3
p0 = [10] * 3
t0 = [20]
deg_rate0 = [0.2] * 3
mu0 = [0.1] # [np.log(2)/(28/10)]
T0 = [10]
n0 = [4]

x0 = np.array(alpha0 + beta0 + K0 + p0 + deg_rate0)
x0 = np.log(x0)

#gamma = np.zeros((3,))
gamma = 0.5
alpha = np.zeros((3,))
beta = np.zeros((3,))
K = np.zeros((3,))
p0 = np.zeros((3,))
t = np.arange(nt)
res = least_squares(residuals(sprof, t), x0=x0) #, diff_step=[1e-3]*len(x0))
#print(res)
alpha = np.exp(res.x[:3])
beta = np.exp(res.x[3:6])
K = np.exp(res.x[6:9])
p0 = np.exp(res.x[9:12])
#t0 = np.exp(res.x[12])
deg_rate = np.exp(res.x[12:15])
#mu0 = np.exp(res.x[13])
#T0 = np.exp(res.x[15])
n = 4 # np.exp(res.x[13])
print(alpha, beta, K, p0, deg_rate)

def deriv(y, t):
    dydt = np.zeros_like(y)
    for idx1 in range(3):
        idx2 = (idx1 + 1) % 3
        gamma = deg_rate[idx2] # mu0 * np.exp(-t/t0) + deg_rate # mu0 / ( 1 + np.exp( (t-t0)*mu0/2 ) ) + deg_rate
        dydt[idx2] = (alpha[idx1] + beta[idx1] * ((y[idx1] / K[idx1])**n)) / (1 + (y[idx1] / K[idx1])**n) - gamma * y[idx2]
    return dydt
t = np.linspace(0,nt*4,360)
model = odeint(deriv, p0, t)
data = sprof

colors = ['b', 'g', 'r']
for c in range(3):
    plt.plot(data[:,c]/K[c], color=colors[c])
    plt.plot(t, model[:,c] / K[c], '--', color=colors[c])

plt.show()

plt.subplot(1, 3, 1)
plt.plot(data[:,0], data[:,1])
plt.plot(model[:,0], model[:,1], '--')
plt.subplot(1, 3, 2)
plt.plot(data[:,0], data[:,2])
plt.plot(model[:,0], model[:,2], '--')
plt.subplot(1, 3, 3)
plt.plot(data[:,1], data[:,2])
plt.plot(model[:,1], model[:,2], '--')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.plot(model[:,0], model[:,1], model[:,2], '--')
plt.plot(data[:,0], data[:,1], data[:,2], '-')
plt.show()
