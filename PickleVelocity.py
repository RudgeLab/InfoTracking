import cPickle
import numpy as np
import matplotlib.pyplot as plt

def pos2pixel(ps,factr):
    return factr*ps

def Pickledx(cs1,cs2,dt,fact): #cellstate1,cellstate2,dt,scalation factor
    dx,arraypos1,arraypos2 = [],[],[]
    i = 0
    ids1 = np.array([cell1.id for (id,cell1) in cs1.iteritems()])
    ids2 = np.array([cell.id for (id,cell) in cs2.iteritems()])
    for (id,cell) in cs1.iteritems():
        try:
            #pos1,pos2 = np.array(cell.pos), np.array(cs2[id].pos)
            pos1,pos2 =  pos2pixel(np.array(cell.pos),fact), pos2pixel(np.array(cs2[id].pos),fact)
            dx.append(pos2-pos1)
            arraypos1.append(pos1)
            arraypos2.append(pos2)
        except KeyError:
            #print 'Data2 has no id',id
            i+=1
    dx,arraypos1,arraypos2 = np.array(dx),np.array(arraypos1),np.array(arraypos2)
    print 'Skipped', i,'ids, which should match this number:', len(ids2)-len(ids1)
    return [dx/dt, arraypos1, arraypos2]

def sumtogrid1(posit,vels,grid,gx,gy,dgx,dgy,a):
    x,y = posit[0],posit[1]
    try:
        xchk,ychk = xycheck(x,y,grid,gx,gy,dgx,dgy)
        grid[xchk,ychk] += np.array([vels[0],vels[1],1,0,0])
    except TypeError:
       a+=1
    return grid,a

def xycheck(x,y,grid,gx,gy,dgx,dgy):
    for ix in range(gx):
        for iy in range(gy):
            dgridx,dgridy = grid[ix,iy,3],grid[ix,iy,4]
            if x >= dgridx and y >= dgridy and x <= dgridx+dgx and y <= dgridy+dgy:
                return ix,iy
#method 2 sum2grid
def sumtogrid2(posit,vels,grid,dgx,dgy):
    x,y = posit[0],posit[1]
    xchk,ychk = x/dgx, y/dgy
    grid[xchk,ychk] += np.array([vels[0],vels[1],1,0,0])
    return grid
def fname2pickle(fname):
    if fname.endswith(".png") or fname.endswith(".jpg"):
        newfname = fname[:len(fname)-4]+".pickle"
    elif fname.endswith(".tiff"):
        newfname = fname[:len(fname)-5]+".pickle"
    return newfname


def main(fname, startframe, nframes, dt, gridfac, worldsize=250.0, forwards = True, GridMethod = None ):
       
    fname2 = fname2pickle(fname)
    
    if forwards == True:
        data = np.array([cPickle.load(open(fname2%(startframe+i*dt))) for i in range(nframes)]) #forward
        imgs = np.array([plt.imread(open(fname%(startframe+i*dt))) for i in range(nframes)]) #forward
    elif forwards == False:
        data = np.array([cPickle.load(open(fname2%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards
        imgs = np.array([plt.imread(open(fname%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards
    
    csa = np.array([element['cellStates'] for element in data])

    worldsize = 250.0
    imagesize = imgs[0].shape[0]
    resizing= imagesize/worldsize 
    C = imagesize/2 
    
    
    print "worldsize = ", worldsize
    print "imagesize = ", imagesize
    print "resizing factor = ", resizing
    
    
    velpos = np.array([Pickledx(csa[i],csa[i+1],dt,resizing) for i in range(nframes-1)]) #dim (nframes-1,3) with dx,x1,x2
    
    
    for element in velpos:
        element[1] = np.array([[C+a,C-b,c] for (a,b,c) in element[1]])
        element[2] = np.array([[C+a,C-b,c] for (a,b,c) in element[2]])
  
    print("Done calculating velocities")
    
    if GridMethod == None:
        GridMethod = input("Select grid method (1) for tracking grid or (2) for reset grid: ") 
        
    gx,gy = int(np.floor(imagesize/gridfac)),int(np.floor(imagesize/gridfac)) #grid dimensions
    print "Grid dimensions: ",gx,gy
    dgx,dgy = gridfac,gridfac
    grid = np.zeros((nframes,gx,gy,5)) #5: vx,vy,number of cells (for normalization), posgridx, posgridy
        
    if GridMethod == 1:   
        for ix in range(gx):
            for iy in range(gy):
                grid[0,ix,iy,:] = [0,0,0,ix*dgx,iy*dgy]
        for step in range(nframes-1):
            poscheck1 = velpos[step,1] 
            vel2add = velpos[step,0]
            lostc=0
            for i in range(len(poscheck1)):
                grid[step],lostc = sumtogrid1(poscheck1[i],vel2add[i],grid[step],gx,gy,dgx,dgy,lostc)
                
            for ix in range(gx):
                for iy in range(gy):
                    grid[step+1,ix,iy] = [0,0,0,grid[step,ix,iy,3],grid[step,ix,iy,4]]
                    if grid[step,ix,iy,2] != 0:
                        grid[step,ix,iy,0] = grid[step,ix,iy,0]/grid[step,ix,iy,2]
                        grid[step,ix,iy,1] = grid[step,ix,iy,1]/grid[step,ix,iy,2]
                        grid[step+1,ix,iy] += [0,0,0,grid[step,ix,iy,0]*dt,-grid[step,ix,iy,1]*dt] #the negative may only need to be for the plots, check

            print "There are",lostc,"cells in the gaps between grid cells in step",step
                
    elif GridMethod == 2:   
        for ix in range(gx):
            for iy in range(gy):
                grid[:,ix,iy,:] = [0,0,0,ix*dgx,iy*dgy]
        for step in range(nframes-1):
            poscheck1 = (velpos[step,1]).astype(int) #method2
            vel2add = velpos[step,0]
        
            for i in range(len(poscheck1)):
                grid[step] = sumtogrid2(poscheck1[i],vel2add[i],grid[step],dgx,dgy)
                
            for ix in range(gx):
                for iy in range(gy):
                    if grid[step,ix,iy,2] != 0:
                        grid[step,ix,iy,0] = grid[step,ix,iy,0]/grid[step,ix,iy,2]
                        grid[step,ix,iy,1] = grid[step,ix,iy,1]/grid[step,ix,iy,2]
    gridstuff = [gx,gy,dgx,dgy]
    print("Done")
    return (grid, gridstuff,velpos,imgs)
