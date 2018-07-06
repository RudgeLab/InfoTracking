import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import infotheory as IT

#for a single gridsquare in all times calculate average X
#Calculate histogram
#Input into infotheory.py


x,y,t = grid.gx/2,grid.gy/2,grid.nframes/2
idd = grid[t,x,y].cells.keys()[0]
for item in vars(grid[t,x,y].cells[idd]):
    print "Calculating entropy of "+item
    
    if type(getattr(grid[t,x,y].cells[idd],item)) == int or type(getattr(grid[t,x,y].cells[idd],item)) == float:
        
        calc_average_all(grid,attribute)
        calc_entropy_all(grid,attribute,100,0)
        
calc_average_all(grid,attribute)
calc_entropy_all(grid,'vx',100,0)
