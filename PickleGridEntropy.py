import PickleVelocity as PV
import numpy as np

def FullPickledx(cs1,cs2,dt,fact,center):
    i = 0
    cells_frame = np.array([])
    ids1 = np.array([cell1.id for (id,cell1) in cs1.iteritems()])
    ids2 = np.array([cell.id for (id,cell) in cs2.iteritems()])
    for (id,cell) in cs1.iteritems():
        try:
            cs1[id].pos[0] = PV.pos2pixel(np.array(cell.pos),fact)[0]+center
            cs1[id].pos[1] = -(PV.pos2pixel(np.array(cell.pos),fact)[1])+center
            cs2[id].pos = cs2[id].pos #just to make it fail
            cells_frame = np.append(cells_frame,cs1)
        except KeyError:
            #print 'Data2 has no id',id
            i+=1
    print 'Skipped', i,'ids, which should match this number:', len(ids2)-len(ids1)
    return cells_frame

def Histogram(grid,Index,nbins):
    np.histogram(np.ravel(grid[Index]),bins = nbins)
    
def ROI_velpos_cell(Method,gridcell,):
    gridcell = []
    if Method == 1:
        ak = 0
    #poner aca valores velocidad y posicion DE CADA celula en una grilla de grid
    #luego lo transformas a histograma2d   ak = 0
def datacheck(neogrid,cell,gx,gy,dgx,dgy):
    x = cell.pos[0]
    y = cell.pos[1]
    for ix in range(gx):
        for iy in range(gy):
            dgridx, dgridy = neogrid[ix,iy,1], neogrid[ix,iy,2]
            if x >= dgridx and y >= dgridy and x <= dgridx+dgx and y <= dgridy+dgy:
                return ix,iy
            
def frame2grid(neogrid,dataframe,gx,gy,dgx,dgy,counter): 
    for cell in dataframe:
        try:
            px,py = datacheck(neogrid,cell,gx,gy,dgx,dgy)
            neogrid[px,py,3] = np.append(neogrid[px,py,3],cell)
        except TypeError:
            counter += 1
        return neogrid,counter
        
def main(fname,startframe,nframes,dt,gridfac,worldsize = 250.0, forwards = True, GridMethod = None):
    grid, gridstuff,velpos,imgs,data,csa = PV.main(fname, startframe, nframes, dt, gridfac, forwards=forwards, GridMethod=GridMethod, PGE = True)
    gx,gy,dgx,dgy = gridstuff[0],gridstuff[1],gridstuff[2],gridstuff[3]
    neogrid = np.zeros((nframes,gx,gy,4))
    
    for frame in range(len(neogrid)):
        for ix in range(gx):
            for iy in range(gy):
                neogrid[frame,ix,iy,:] = [grid[frame,ix,iy,2],grid[frame,ix,iy,3],grid[frame,ix,iy,4],0] #only keep position of grid and number of cells in grid
                aux = []
                neogrid[frame,ix,iy,3] = aux
    imagesize = imgs[0].shape[0]
    resizing= imagesize/worldsize 
    C = imagesize/2     
    
    dataframes = np.array([FullPickledx(csa[i],csa[i+1],dt,resizing,C) for i in range(nframes-1)]) #dim (nframes-1,3) with dx,x1,x2
    
    for frame in dataframes:
        neogrid[frame],counter = frame2grid(neogrid[frame],dataframes,gx,gy,dgx,dgy)
        print "There are",counter,"cells in the gaps between grid cells in step",frame

    return neogrid



afname = "/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-%04d0.png"
astartframe = 40
anframes = 5
adt = 1 #There's a bit of trouble with this
agridfactor = 32 #pixels per grid
aneogrid = main(afname,astartframe,anframes,adt,agridfactor,forwards = True, GridMethod = 1)