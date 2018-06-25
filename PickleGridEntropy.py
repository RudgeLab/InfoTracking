import cPickle
import numpy as np
import matplotlib.pyplot as plt

class Grid():
    def __init__(self,nframes,gx,gy,dgx,dgy,center,resizing):
        self.grid = []
        self.nframes = nframes
        self.frames = np.arange(0,nframes,1)
        self.gx = gx
        self.gy = gy
        self.dgx = dgx
        self.dgy = dgy
        self.center = center
        self.resize = resizing
        for frame in range(nframes):
            frame1 = []
            for ix in range(gx):
                for iy in range(gy):
                    frame1.append(Ensemble(frame,ix,iy,ix*dgx,iy*dgy))
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
        
    def getentropy(self,ix,iy,attribute):
        entropy_t_list = []
        for it in range(self.nframes):
            if len(self.grid[it][ix*self.gx+iy].entropy)!= 0:
                entropy_t = self.grid[it][ix*self.gx+iy].entropy[attribute]
            else:
                entropy_t = 0
            entropy_t_list.append(entropy_t)
        return entropy_t_list
            
      
        
class Ensemble():
    def __init__(self,frame,ix,iy,px0,py0):
        self.px = px0
        self.py = py0
        self.vx = 0
        self.vy = 0
        self.ix = ix
        self.iy = iy
        self.t1 = frame
        self.cells = {}
        self.skipped = 0
        self.actualcells = 0
        self.entropy = {}
        self.averages = {}

    def addCell(self,cell,id): #cell = cellstate
        self.cells[id] = cell

    def CalcVel(self,nextstepcells,lineage,factor,dt):
        dx,dy = 0,0
        for id,next_cell in nextstepcells.iteritems():
            dx_cell = 0
            dy_cell = 0
            try:
                dx_cell = next_cell.pos[0]-self.cells[id].pos[0]
                dy_cell = next_cell.pos[1]-self.cells[id].pos[1]
                self.actualcells +=1
            except KeyError:
                # Previous cell does not exist, use parent cell
                pid = lineage[id]
                dx_cell = next_cell.pos[0]-self.cells[pid].pos[0]
                dy_cell = next_cell.pos[1]-self.cells[pid].pos[1]
                self.skipped +=1
                self.actualcells += 0.5 # Count as 1/2 to take average of children
            dx += dx_cell
            dy += dy_cell
            
        if self.actualcells != 0:
            dx = dx/self.actualcells
            dy = dy/self.actualcells
        self.averages['vel'] = [factor*dx/dt,-factor*dy/dt]
        self.vx = factor*dx/dt
        self.vy = -factor*dy/dt

def fname2pickle(fname):
    if fname.endswith(".png") or fname.endswith(".jpg"):
        newfname = fname[:len(fname)-4]+".pickle"
    elif fname.endswith(".tiff"):
        newfname = fname[:len(fname)-5]+".pickle"
    return newfname

def pos2pixel(ps,factr):
    return factr*ps

def checkcellingrid(cellstate,gsq,resize,dgx,dgy,center):
    x,y = pos2pixel(cellstate.pos[0],resize)+center,-pos2pixel(cellstate.pos[1],resize)+center
    xg,yg = gsq.px,gsq.py
    if x >= xg and x < xg+dgx and y >= yg and y < yg+dgy:
        return True
    else:
        return False

def main(fname,startframe,nframes,dt,gridfac,worldsize = 250.0, forwards = True, GridMethod = None):
    
    fname2 = fname2pickle(fname)    
    if forwards == True:
        data = np.array([cPickle.load(open(fname2%(startframe+(i*dt)))) for i in range(nframes)]) #forward
        imgs = np.array([plt.imread(open(fname%(startframe+(i*dt)))) for i in range(nframes)]) #forward

    elif forwards == False:
        data = np.array([cPickle.load(open(fname2%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards
        imgs = np.array([plt.imread(open(fname%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards
     
    csa = np.array([element['cellStates'] for element in data])
    lina = np.array([element['lineage'] for element in data])
    
    imagesize = imgs[0].shape[0]
    resizing= imagesize/worldsize 
    C = imagesize/2 
    
    
    print "worldsize = ", worldsize
    print "imagesize = ", imagesize
    print "resizing factor = ", resizing  
    gx,gy = int(np.floor(imagesize/gridfac)),int(np.floor(imagesize/gridfac)) #grid dimensions
    print "Grid dimensions: ",gx,gy
    dgx,dgy = gridfac,gridfac
    
    grid = Grid(nframes,gx,gy,dgx,dgy,C,resizing)

    for t in range(nframes-1):
        #Add cells to gridcells :
        for ix in range(gx):
            for iy in range(gy):
                for (id,cell) in csa[t].iteritems():
                    if checkcellingrid(cell,grid[t,ix,iy],resizing,dgx,dgy,grid.center) == True:
                        grid[t,ix,iy].addCell(cell,id)
        #calc velocity of grid at t = 0 to t+1:
        skip,cellno,actual = 0,0,0
        for ix in range(gx):
            for iy in range(gy):
                grid[t,ix,iy].CalcVel(csa[t+1],lina[t+1],resizing,dt) 
                skip,cellno,actual = skip+grid[t,ix,iy].skipped,cellno+len(grid[t,ix,iy].cells),actual+grid[t,ix,iy].actualcells
                #Should we remove skipped cells? might affect entropy flux calculation  
                #Move grids t+1:
                grid[t+1,ix,iy].px = grid[t,ix,iy].px + grid[t,ix,iy].vx*dt
                grid[t+1,ix,iy].py = grid[t,ix,iy].py + grid[t,ix,iy].vy*dt
        print 'Skipped', skip,'ids, which should match this number:', cellno-actual
     
         
    return grid,csa,imgs

def squareplot(x,y,dgx,dgy):
    plt.plot([x,x+dgx,x+dgx,x,x],[y,y,y+dgy,y+dgy,y],"k-",linewidth = 0.5)
def plotensemble(grid,t,gx,gy,dgx,dgy):
    for ix in range(gx):
        for iy in range(gy):
            if grid[t,ix,iy].actualcells != 0:
                squareplot(grid[t,ix,iy].px,grid[t,ix,iy].py,dgx,dgy)
                
def plott(grid,imgs,gx,gy,dgx,dgy):
    for i in range(len(grid.frames)-1):
        plt.clf()
        plt.hold(True)
        plt.imshow(imgs[i],origin = 'lower')
        plotensemble(grid,i,gx,gy,dgx,dgy)
        plt.hold(False)
        plt.pause(2)
       

afname = "/Users/timrudge/cellmodeller/data/testing-18-06-25-01-09/step-%05d.png"
astartframe = 10
anframes = 20
adt = 1 #There's a bit of trouble with this
agridfactor = 64 #pixels per grid
grid,cs,ims = main(afname,astartframe,anframes,adt,agridfactor,forwards = False, GridMethod = 1)
