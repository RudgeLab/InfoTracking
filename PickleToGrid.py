import numpy as np
import cPickle
#import matplotlib.pyplot as plt

class Grid():
    def __init__(self,nframes,gx,gy,dgx,dgy,center,resizing,dt,forwards):
        self.grid = []
        self.nframes = nframes
        self.frames = np.arange(0,nframes,1)
        self.gx = gx
        self.gy = gy
        self.dgx = dgx
        self.dgy = dgy
        self.center = center
        self.resize = resizing
        self.dt = dt
        self.forwards = forwards
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
     
    def timestep(self,t,cellstate,lineage):
        print '-'*16,'Step ',t,'-'*16
        #Add cells to gridcells :
        for ix in range(self.gx):
            for iy in range(self.gy):
                for (id,cell) in cellstate[t].iteritems():
                    if self[t,ix,iy].CheckCellInEnsemble(cell,self.resize,self.dgx,self.dgy,self.center) == True:
                        self[t,ix,iy].addCell(cell,id)
        #calc velocity of grid at t = 0 to t+1: 
        counted_total = 0
        for ix in range(self.gx):
            for iy in range(self.gy):
                if self.forwards == True:
                    Total,cell_no = self[t,ix,iy].CalcVel(cellstate[t],cellstate[t+1],lineage[t+1],self.dt,self.resize,self.dgx,self.dgy,self.center,self.forwards)
                    
                elif self.forwards == False:
                    Total,cell_no = self[t,ix,iy].CalcVel(cellstate[t],cellstate[t+1],lineage[t-1],self.dt,self.resize,self.dgx,self.dgy,self.center,self.forwards)
                
                counted_total += cell_no
                #Should we remove skipped cells? might affect entropy flux calculation  
                #Move grids t+1:
                self[t+1,ix,iy].px = self[t,ix,iy].px + self[t,ix,iy].vx*self.dt
                self[t+1,ix,iy].py = self[t,ix,iy].py + self[t,ix,iy].vy*self.dt
        if self.forwards == True:
            if Total - counted_total != 0:
                print 'Cells in the void between ensembles: ', Total - counted_total
        if self.forwards == False:
            if Total > counted_total:
                print 'Cells in the void between ensembles: ', Total - counted_total
        print 'Done'
         
    
        
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
        self.cell_number = 0
        self.entropy = {}
        self.averages = {}

    def addCell(self,cell,id): #cell = cellstate
        #print 'added cell: ',id, 'to', (self.px,self.py)
        self.cells[id] = cell
        
    def CalcVel(self,cellstate,nextstepcells,lineage,dt,resize,dgx,dgy,center,forwards):
        dx,dy = 0,0
        total_cells = 0
        for id,next_cell in nextstepcells.iteritems():
            dx_cell = 0
            dy_cell = 0
            total_cells += 1
            if self.CheckCellInEnsemble(next_cell,resize,dgx,dgy,center) == True:
                
                #print 'Calculating velocity of Ensemble: ', (self.px, self.py),' cell id ',id
                try:
                    dx_cell = next_cell.pos[0]-cellstate[id].pos[0]
                    dy_cell = next_cell.pos[1]-cellstate[id].pos[1]
                    self.cell_number += 1
                    #print '+++++Success'
                except KeyError:
                    if forwards == True:
                        #print '----- Cell not in current step, checking for division'
                        # Previous cell does not exist, use parent cell
                        pid = lineage[id]
                        #print 'Using parent cell: ',pid
                        dx_cell = next_cell.pos[0]-cellstate[pid].pos[0]
                        dy_cell = next_cell.pos[1]-cellstate[pid].pos[1]
                        self.cell_number += 1 # Count as 1/2 to take average of children
                    if forwards == False:
                        #print '----- Cell not in current step, checking for division'
                        # Previous cell does not exist, use parent cell
                        #print lineage
                        pids = [key for key, value in lineage.iteritems() if value == id]
                        pid = pids[0]
                        #print 'Using daughter cell: ',pid
                        dx_cell = next_cell.pos[0]-cellstate[pid].pos[0]
                        dy_cell = next_cell.pos[1]-cellstate[pid].pos[1]
                        self.cell_number += 1 # Count as 1/2 to take average of children
                dx += dx_cell
                dy += dy_cell
                        
        if self.cell_number != 0:
            dx = dx/self.cell_number
            dy = dy/self.cell_number
        self.averages['vel'] = [resize*dx/dt,-resize*dy/dt]
        self.vx = resize*dx/dt
        self.vy = -resize*dy/dt
        return total_cells,self.cell_number
    def CheckCellInEnsemble(self,cellstate,resize,dgx,dgy,center):
        x,y = pos2pixel(cellstate.pos[0],resize)+center,-pos2pixel(cellstate.pos[1],resize)+center
        xg,yg = self.px, self.py
        if x >= xg and x < xg+dgx and y >= yg and y < yg+dgy:
            return True
        else:
            return False
'''
def fname2pickle(fname):
    if fname.endswith(".png") or fname.endswith(".jpg"):
        newfname = fname[:len(fname)-4]+".pickle"
    elif fname.endswith(".tiff"):
        newfname = fname[:len(fname)-5]+".pickle"
    return newfname
'''
def pos2pixel(ps,factr):
    return factr*ps

def loadPickle(fname,startframe,nframes,dt, forwards):
     
    if forwards == True:
        data = np.array([cPickle.load(open(fname%(startframe+(i*dt)))) for i in range(nframes)]) #forward
    elif forwards == False:
        data = np.array([cPickle.load(open(fname%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards     

    cellstate = np.array([element['cellStates'] for element in data])
    lineage = np.array([element['lineage'] for    element in data])
    return cellstate,lineage

def main(fname,startframe,nframes,dt,gridfac,worldsize = 250.0, forwards = True, GridMethod = None):
    
    cellstate, lineage = loadPickle(fname,startframe,nframes,dt,forwards)

    resizing = 1
    C = worldsize/2 
    gx,gy = int(np.floor(worldsize/gridfac)),int(np.floor(worldsize/gridfac)) #grid dimensions
    dgx,dgy = gridfac,gridfac
    
    print "worldsize = ", worldsize
    print "resizing factor = ", resizing  
    print "Grid dimensions: ",gx,gy
    
    grid = Grid(nframes,gx,gy,dgx,dgy,C,resizing,dt,forwards)
    
    for it in range(1,nframes-1):
        grid.timestep(it,cellstate,lineage)
        
    return grid,cellstate



afname = "/Users/Medina/cellmodeller/data/Practice_Script_Log2-18-08-08-16-11/step-%05d.pickle"
astartframe = 0
anframes = 180
adt = 1 #There's a bit of trouble with this
agridfactor = 16 #pixels per grid
grid,cs = main(afname,astartframe,anframes,adt,agridfactor,forwards = False, GridMethod = 1)
