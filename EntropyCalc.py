#importa grid de PGE
#histograma de velocidades de cada gridcell para todos los t
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

def mainEC(grid,AV):
    times = []
    for t in range(len(grid.frames)):
        variable = []
        for ix in range(grid.gx):
            for iy in range(grid.gy):
                
                if AV == 'vel':
                    AV = ['vx','vy']
                    vx = getattr(grid[t,ix,iy],AV[0])
                    vy = getattr(grid[t,ix,iy],AV[1])
                    elem = np.sqrt(((vx)**2)+(vy)**2)
                    
                if AV == 'x':
                    aux = []
                    for id in grid[t,ix,iy].cells.keys():
                        var = getattr(grid[t,ix,iy].cells[id],'x')
                        aux.append(var)
                    elem = np.ravel(np.array(aux))
                
                variable.append(elem)
            times.append(variable)     
    return times


def alltosingle(variab):
    variab = np.array(variab)
    variab = np.ravel(variab)
    return variab

def makehist(allvars,nbins,skip):
    plt.hist(allvars[allvars >=skip],bins = nbins,normed = 1,edgecolor='#E6E6E6', color='#EE6666')
             
    
def coolhist():
    
    #ax = plt.axes(axisbg='#E6E6E6')
    ax = plt.axes()
    ax.set_axisbelow(True)
    plt.grid(color='w', linestyle='solid')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(colors='gray', direction='out')
    for tick in ax.get_xticklabels():
        tick.set_color('gray')
    for tick in ax.get_yticklabels():
        tick.set_color('gray')
        
def fitcurve(allvars,nbins,skip):
    (mu, sigma) = norm.fit(allvars[allvars >=skip])
    n, bins, patches =  plt.hist(allvars[allvars >=skip],bins = nbins,normed = 1,edgecolor='#E6E6E6', color='#EE6666')
    y = mlab.normpdf( bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2)
    plt.show()
    
alltimes = mainEC(grid,'vel')
allvars = alltosingle(alltimes)
#coolhist()
fitcurve(allvars,15,0)
makehist(allvars,15,0)
'''
for i in range(len(alltimes)):
    plt.clf()
    plt.hold(True)
    fitcurve(alltimes[i],15,0.05)
    plt.show()
    plt.hold(False)
    plt.pause(0.5)
'''