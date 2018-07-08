import cPickle
import numpy as np
import matplotlib.pyplot as plt
import infotheory as IT

class Grid():
    def __init__(self,nframes,gx,gy,dgx,dgy,center,resize,dt):
        self.grid = []
        self.nframes = nframes
        self.frames = np.arange(0,nframes,1)
        self.gx = gx
        self.gy = gy
        self.dgx = dgx
        self.dgy = dgy
        self.center = center
        self.resize = resize
        self.dt = dt
        for frame in range(nframes):
            frame1 = []
            for ix in range(gx):
                for iy in range(gy):
                    frame1.append(Ensemble(frame,ix,iy,ix*dgx,iy*dgy,self.resize))
            self.grid.append(frame1)
            
    def __getitem__(self,i):
        
        if type(i) == int:
            return self.grid[i]

        if len(i) == 3 and type(i[0]) == int:
            frame,ix,iy = i
            return self.grid[frame][ix*self.gx+iy]
        
        if len(i) == 3 and type(i[0]) == str:
            attribute,ix,iy = i
            entropy_t_list = []
            for it in range(self.nframes):
                if len(self.grid[it][ix*self.gx+iy].entropy)!= 0:
                    entropy_t = self.grid[it][ix*self.gx+iy].entropy[attribute]
                else:
                    entropy_t = 0
                entropy_t_list.append(entropy_t)
            return entropy_t_list
        
    def add_cells_to_ensembles(self,csa,t):
        for ix in range(self.gx):
            for iy in range(self.gy):
                for (id,cell) in csa[t].iteritems():
                    if self[t,ix,iy].checkcellingrid(cell,self.resize,self.dgx,self.dgy,self.center) == True:
                    #if checkcellingrid(cell,grid[t,ix,iy],resizing,dgx,dgy,grid.center) == True:
                        print "adding cell"
                        self[t,ix,iy].addCell(cell)
                        
    def calc_velocities_of_ensembles(self,cellstates,lineage,t):
        skip,cellno,actual = 0,0,0
        for ix in range(self.gx):
            for iy in range(self.gy):
                if len(self[t,ix,iy].cells) != 0:
                    self[t,ix,iy].CalcVel(cellstates[t], cellstates[t+1], lineage[t+1], self.resize, self.dt) 
                    skip,cellno,actual = skip+self[t,ix,iy].skipped,cellno+len(self[t,ix,iy].cells),actual+self[t,ix,iy].actualcells
                    self[t+1,ix,iy].px = self[t,ix,iy].px + self[t,ix,iy].vx*self.dt
                    self[t+1,ix,iy].py = self[t,ix,iy].py + self[t,ix,iy].vy*self.dt
        print t,'- Skipped', skip,'ids, which should match this number:', cellno-actual
    
    def calc_average_all(self,attribute):
        for it in range(self.nframes):
            for ix in range(self.gx):
               for iy in range(self.gy):
                   if len(self[it,ix,iy].cells) != 0:
                       self[it,ix,iy].calculate_average(attribute)
                       
    def calc_entropy_all(self,attribute,nbins,skip):           
        for ix in range(self.gx):
            for iy in range(self.gy):
                for it in range(self.nframes):
                    self[it,ix,iy].entropy_calc(attribute,nbins,skip)
                
    def getentropy(self,ix,iy,attribute):  
        entropy_t_list = []
        for it in range(self.nframes):
            if len(self.grid[it][ix*self.gx+iy].entropy)!= 0:
                entropy_t = self.grid[it][ix*self.gx+iy].entropy[attribute]     
                
            else: 
                entropy_t = 0
            entropy_t_list.append(entropy_t)
            
        return np.array(entropy_t_list)
    
    def getaverage(self,ix,iy,attribute):
        avg_t_list = []
        
        for it in range(self.nframes):
            if len(self.grid[it][ix*self.gx+iy].averages)!= 0:
                avg_t = self.grid[it][ix*self.gx+iy].averages[attribute]
                
            else:
                avg_t = 0
                
            avg_t_list.append(avg_t)
            
        return np.array(avg_t_list)
    
    def plot_ensembles(self,t,gx,gy,dgx,dgy):
        for ix in range(gx):
            for iy in range(gy):
                if grid[t,ix,iy].actualcells != 0:
                    grid[t,ix,iy].plot_ensemble(grid.dgx,grid.dgy)
   
    def pos2pixel(self,value_in_pixels):
        return self.resize * value_in_pixels
    
    def pixel2pos(self,value_in_rw):
        return value_in_rw / self.resize
    
 #-----------------------------------------------------------------------------
       
class Ensemble():
    
    def __init__(self,frame,ix,iy,px0,py0,resize):
        self.px = px0
        self.py = py0
        self.vx = 0
        self.vy = 0
        self.ix = ix
        self.iy = iy
        self.t1 = frame
        self.cells = {}
        self.entropy = {}
        self.averages = {}
        self.skipped = 0
        self.actualcells = 0
        self.resize = resize
        
    def addCell(self,cell): #cell = cellstate
        id = cell.id
        print "adding cell ", id
        self.cells[id] = cell
        self.cells[id].vx = 0
        self.cells[id].vy = 0

    def CalcVel(self,cellstate,nextstepcells,lineage,factor,dt):
        dx,dy = 0,0
        
        print "Lineage: "
        print lineage
        
        print "cells "
        print self.cells
        
        print "nextstepcells "
        print nextstepcells
    
        for id,next_cell in nextstepcells.iteritems():
            dx_cell = 0
            dy_cell = 0
            
            if id in self.cells.keys():
                dx_cell = next_cell.pos[0]-self.cells[id].pos[0]
                dy_cell = next_cell.pos[1]-self.cells[id].pos[1]
                self.cells[id].vx = dx_cell
                self.cells[id].vy = dy_cell
                self.actualcells +=1
            else:
                # Previous cell does not exist, use parent cell
                print("Cell ",id)
                pid = lineage[id]
                print "pid ", pid
                dx_cell = next_cell.pos[0]-cellstate[pid].pos[0]
                dy_cell = next_cell.pos[1]-cellstate[pid].pos[1]
                self.cells[pid].vx = dx_cell
                self.cells[pid].vy = dy_cell
                self.actualcells += 1 # Count as 1/2 to take average of children
                
            dx += dx_cell
            dy += dy_cell
            
        if self.actualcells != 0:
            dx = dx/self.actualcells
            dy = dy/self.actualcells
        self.averages['vx'] = factor*dx/dt
        self.averages['vy'] = -factor*dy/dt
        self.vx = factor*dx/dt
        self.vy = -factor*dy/dt
        print 'vx: ', self.vx, 'vy: ', self.vy
        
    def calculate_average(self,attribute):
        avg = 0
        n = 0
        for id in self.cells.keys():
            cell_atr = getattr(self.cells[id],attribute, None)
            if cell_atr:
                avg += cell_atr
                n += 1
            self.averages[attribute] = avg/n
    
    def checkcellingrid(self,cellstate,resize,dgx,dgy,center):
        
        x,y = self.pos2pixel(cellstate.pos[0]) + center,- self.pos2pixel(cellstate.pos[1]) + center #cell position in transformed to pixels
        xg,yg = self.px,self.py
        
        if x >= xg and x < xg+dgx and y >= yg and y < yg+dgy:
            return True
        else:
            return False
        
    def entropy_calc(self,attribute,nbins,skip):
        if self.actualcells != 0:
            histogram,attr_edges,attrib_list = self.histogram_ensemble(attribute,nbins,skip)
            self.entropy[attribute] = IT.entropy(histogram)
  
    def histogram_ensemble(self,attribute,nbins,skip):
        attrib_list = [getattr(self.cells[id],attribute,np.nan) for id in self.cells.keys()]
        attrib_arr = np.array(attrib_list)
    
        attrib_arr = attrib_arr[~np.isnan(attrib_arr)]
        hist,attr_edge = np.histogram(attrib_arr[attrib_arr>=skip],bins = nbins)
        
        return hist,attr_edge,attrib_list

    def pos2pixel(self,value_in_pixels):
        return self.resize * value_in_pixels
    
    def pixel2pos(self,value_in_rw):
        return value_in_rw / self.resize
    
    def plotensemble(self,dgx,dgy):
        if self.actualcells != 0:
            self.squareplot(self.px,self.py,dgx,dgy)
                    
    def squareplot(x,y,dgx,dgy):
        plt.plot([x,x+dgx,x+dgx,x,x],[y,y,y+dgy,y+dgy,y],"k-",linewidth = 0.5)
    
    
 #-----------------------------------------------------------------------------   
 
#File reading function:
        
def fname2pickle(fname):
    if fname.endswith(".png") or fname.endswith(".jpg"):
        newfname = fname[:len(fname)-4]+".pickle"
    elif fname.endswith(".tiff"):
        newfname = fname[:len(fname)-5]+".pickle"
    return newfname

#Load Data
    
def LoadData(fname,startframe,nframes,dt,forwards = True):
    
    fname2 = fname2pickle(fname)  
    
    if forwards == True:
        data = np.array([cPickle.load(open(fname2%(startframe+(i*dt)))) for i in range(nframes)]) #forward
        imgs = np.array([plt.imread(open(fname%(startframe+(i*dt)))) for i in range(nframes)]) #forward

    elif forwards == False:
        data = np.array([cPickle.load(open(fname2%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards
        imgs = np.array([plt.imread(open(fname%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards
       
    cellstates = np.array([element['cellStates'] for element in data])
    print "cellstates"
    print cellstates[0]
    lineage = np.array([element['lineage'] for element in data])
    
    return data,imgs,cellstates,lineage

#Image parameters
    
def set_image_parameters(imgs,worldsize):
    imagesize = imgs[0].shape[0]
    resizing= imagesize/worldsize 
    center = imagesize/2 
    
    print "worldsize = ", worldsize
    print "imagesize = ", imagesize
    print "resizing factor = ", resizing 
    
    return imagesize, resizing, center

#Grid parameters based on image parameters
    
def set_grid_parameters(image_size,gridfac):
    
    gx,gy = int(np.floor(image_size/gridfac)),int(np.floor(image_size/gridfac)) #grid dimensions
    
    print "Grid dimensions: ",gx,gy
    
    dgx,dgy = gridfac,gridfac
    
    return gx,gy,dgx,dgy

#Extra plotting functions:
    
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

#Main

def main(fname,startframe,nframes,dt,gridfac,worldsize = 250.0, forwards = True, GridMethod = None):
    data, imgs, cellstates, lineage = LoadData(fname,startframe,nframes,dt,forwards = True)
    
    image_size, resize, center = set_image_parameters(imgs,worldsize)
    
    gx,gy,dgx,dgy = set_grid_parameters(image_size,gridfac)

    grid = Grid(nframes,gx,gy,dgx,dgy,center,resize,dt)
    
    for t in range(nframes-1):
        print "*** Frame = ", t
        #Add cellstates to grid at time t :
        
        grid.add_cells_to_ensembles(cellstates,t)
        
        #calc velocity of grid at t = 0 to t+1:
        grid.calc_velocities_of_ensembles(cellstates,lineage,t)
        
    return grid,cellstates,imgs
                
def plott(grid,imgs,gx,gy,dgx,dgy):
    for i in range(len(grid.frames)-1):
        plt.clf()
        plt.hold(True)
        plt.imshow(imgs[i],origin = 'lower',cmap='gray')
        grid.plotensembles(i,gx,gy,dgx,dgy)
        plt.hold(False)
        plt.pause(2)
       
#File test parameters
afname = "/home/timrudge/cellmodeller/data/info_tracking-18-06-08-12-59/step-%05d.png"
#afname = "/Users/Ignacio/cellmodeller/data/IICProject-18-06-26-18-10/step-%05d.png"
astartframe = 0
anframes = 20
adt = 2
agridfactor = 64 #pixels per grid
grid,cst,imgs = main(afname,astartframe,anframes,adt,agridfactor,forwards = True, GridMethod = 1)
