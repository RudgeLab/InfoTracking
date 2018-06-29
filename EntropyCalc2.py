import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import infotheory as IT

#for a single gridsquare in all times calculate average X
#Calculate histogram
#Input into infotheory.py

def calculate_average(ensemble,attribute):
    avg = 0
    n = 0
    for id in ensemble.cells.keys():
        cell_atr = getattr(ensemble.cells[id],attribute, None)
        if cell_atr:
            avg += cell_atr
            n += 1
    ensemble.averages[attribute] = avg/len(ensemble.cells)
    
def histogram_ensemble(ensemble,attribute,nbins,skip):
    attrib_list = [getattr(ensemble.cells[id],attribute,np.nan) for id in ensemble.cells.keys()]
    attrib_arr = np.abs(np.array(attrib_list))
    
    attrib_arr = attrib_arr[~np.isnan(attrib_arr)]
    hist,attr_edge = np.histogram(attrib_arr[attrib_arr>=skip],bins = nbins)
    return hist,attr_edge,attrib_list

def entropycalc(grid,attribute,x,y,nbins,skip):
    for t in range(grid.nframes):
        if grid[t,x,y].actualcells != 0:
            histogram,attr_edges,attrib_list = histogram_ensemble(grid[t,x,y],attribute,nbins,skip)
            grid[t,x,y].entropy[attribute] = IT.entropy(histogram)
        
def addentropy(grid,attribute,nbins,skip):
    #attribute = attribute_analysis()
    for it in range(grid.nframes):
        for ix in range(grid.gx):
            for iy in range(grid.gy):
                if len(grid[it,ix,iy].cells) != 0:
                    calculate_average(grid[it,ix,iy],attribute)
    for ix in range(grid.gx):
        for iy in range(grid.gy):
                entropycalc(grid,attribute,ix,iy,nbins,skip)
    
x,y,t = grid.gx/2,grid.gy/2,grid.nframes/2
idd = grid[t,x,y].cells.keys()[0]
for item in vars(grid[t,x,y].cells[idd]):
    print "Calculating entropy of "+item
    if type(getattr(grid[t,x,y].cells[idd],item)) == int or type(getattr(grid[t,x,y].cells[idd],item)) == float:
        addentropy(grid,item,100,0)

def plotentrosin(grid,attrib,x,y,colorcode=None):
    plt.xlabel("log[ Timestep ]")
    plt.ylabel("$e$^[ Entropy of "+attrib+' ]')
    t = np.arange(grid.nframes)*16
    entro = grid.getentropy(x,y,attrib)[::-1]
    if colorcode:
        plt.plot(t,entro,colorcode,linewidth= 0.2)
    else:
        C = grid.gx/2
        if (x-C)**2+(y-C)**2 <= 10:
            colorRGB = (0,0.5,0)
        elif (x-C)**22+(y-C)**22 > 10:
            colorRGB = (0,0,0.5)
        plt.plot(np.log(t),np.exp(entro),color = colorRGB,linewidth= 0.2)
    
def plotentropall(grid,attrib):
    t = np.arange(grid.nframes)*grid.dt
    entro=np.zeros(grid.nframes)
    k = 0
    plt.xlabel("log[ Timestep ]")
    plt.ylabel("$e$^[ Entropy of "+attrib+' ]')
    for ix in range(5,14):
        for iy in range(5,14):
            try:
                entro += grid.getentropy(ix,iy,attrib)[::-1]
                k += 1
            except:
                entro += 0
    plt.plot(np.log(t),np.exp(entro/k),'r--')
                
            
'''    
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

for i in range(len(alltimes)):
    plt.clf()
    plt.hold(True)
    fitcurve(alltimes[i],15,0.05)
    plt.show()
    plt.hold(False)
    plt.pause(0.5)
'''