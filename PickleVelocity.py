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


'''-------MAIN-----'''
plt.ioff()

dt=1
nframes = 35
startframe = 18
 
#fname = "/Users/timrudge/cellmodeller/data/weiner-17-12-07-17-56/step-%04d0.pickle"
fname = "/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-%04d0.pickle"
#fname2 = "/Users/timrudge/cellmodeller/data/weiner-17-12-07-17-56/step-%04d0.png"
fname2 = "/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-%04d0.png"

data = np.array([cPickle.load(open(fname%(startframe+i*dt))) for i in range(nframes)]) #forward
imgs = np.array([plt.imread(open(fname2%(startframe+i*dt))) for i in range(nframes)]) #forward

#data = np.array([cPickle.load(open(fname%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards
#imgs = np.array([plt.imread(open(fname2%(startframe+(nframes-i)*dt))) for i in range(nframes)]) #backwards



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
for item in velpos:
    item[2] = np.array([[C+a,C-b,c] for (a,b,c) in item[2]])
    
print("Done calculating velocities")


#method 1 sum2grid
def sumtogrid1(posit,vels,grid,dgxs,dgys,a):
    x,y = posit[0],posit[1]
    try:
        xchk,ychk = xycheck(x,y,grid)
        grid[xchk,ychk] += np.array([vels[0],vels[1],1,0,0])
    except TypeError:
       a+=1
    return grid,a

def xycheck(x,y,grid):
    for ix in range(gx):
        for iy in range(gy):
            dgridx,dgridy = grid[ix,iy,3],grid[ix,iy,4]
            if x >= dgridx and y >= dgridy and x <= dgridx+dgx and y <= dgridy+dgy:
                return ix,iy
#method 2 sum2grid
def sumtogrid2(posit,vels,grid,dgxs,dgys):
    x,y = posit[0],posit[1]
    xchk,ychk = x/dgxs, y/dgys
    grid[xchk,ychk] += np.array([vels[0],vels[1],1,0,0])
    return grid


GridMethod = input("Select grid method (1) for tracking grid or (2) for reset grid: ") 

gridfac = 40 #pixels per grid
gx,gy = int(np.floor(imagesize/gridfac)),int(np.floor(imagesize/gridfac)) #grid dimensions
print "Grid dimensions: ",gx,gy
dgx,dgy = gridfac,gridfac
grid = np.zeros((nframes,gx,gy,5)) #5: vx,vy,number of cells (for normalization), posgridx, posgridy
    
if GridMethod == 1:   
    for ix in range(gx):
        for iy in range(gy):
            grid[0,ix,iy,:] = [0,0,0,ix*dgx,iy*dgy]
            
    for step in range(nframes-1):
        if step != 0:
            for ix in range(gx):
                for iy in range(gy):
                    grid[step,ix,iy] = [0,0,0,grid[step-1,ix,iy,3],grid[step-1,ix,iy,4]]
        poscheck1 = velpos[step,1] 
        vel2add = velpos[step,0]
        lostc=0
        for i in range(len(poscheck1)):
            grid[step],lostc = sumtogrid1(poscheck1[i],vel2add[i],grid[step],dgx,dgy,lostc)
            
        for ix in range(gx):
            for iy in range(gy):
                if grid[step,ix,iy,2] != 0:
                    grid[step,ix,iy,0] = grid[step,ix,iy,0]/grid[step,ix,iy,2]
                    grid[step,ix,iy,1] = grid[step,ix,iy,1]/grid[step,ix,iy,2]
                    grid[step,ix,iy] += [0,0,0,grid[step,ix,iy,0]*dt,-grid[step,ix,iy,1]*dt] 
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
                    
def velplotter(image,position1,position2,velocity,j):
    #fig1 = plt.figure()
    plt.imshow(image)
    plt.plot(position1[:,0],position1[:,1],"ro",markersize = 0.5)
    plt.quiver(position1[:,0],position1[:,1],velocity[:,0],velocity[:,1])
    #plt.show()
    #fig1.savefig('velstep-00'+str(startframe+j)+'0.pdf')                    
def squareplot(x,y):
    plt.plot([x,x+dgx,x+dgx,x,x],[y,y,y+dgy,y+dgy,y],"k-")
def plotensemble(grid):
    for ix in range(gx):
        for iy in range(gy):
            if grid[ix,iy,2] != 0:
                squareplot(grid[ix,iy,3],grid[ix,iy,4])
def plotter():
    print("Results are ready, show plots? Yes will plot last frame and will include everything ")
    plot = input("(1) Yes, (2) No, (3) Advanced: ")
    F2P = nframes-2
    ens = 1
    velplot = 1
    cent = 1
    if plot == 3:
        print "Max allowed frame is",nframes-2,": "
        F2P = input("Choose frame: ")
        velplot = input("Plot cells and velocity of cells? (1) Yes, (2) No: ")
        ens = input("Plot ensembles and velocity of ensemble? (1) Yes, (2) No: ")
        cent = input("Plot evolution of grid center? (1) Yes, (2) No: ")
      
    if plot != 2:
        #Grid centers plot
        if cent == 1:
            for k in range(nframes-1):
                for x in range(len(grid[k])): 
                    for y in range(len(grid[k])):
                        if grid[k,x,y,2] != 0:
                            plt.plot(grid[k,x,y,3]+gridfac/2,grid[k,x,y,4]+gridfac/2,"bo", markersize = 0.8)
                   
        if ens == 1:
           plotensemble(grid[F2P])
           for x in range(len(grid[F2P])): 
               for y in range(len(grid[F2P])):
                   if grid[F2P,x,y,2] != 0:
                       plt.quiver(grid[F2P,x,y,3]+gridfac/2,grid[F2P,x,y,4]+gridfac/2,grid[F2P,x,y,0],grid[F2P,x,y,1])
                       #plt.text((dgx*x)+gridfac/2,(dgy*y)+gridfac/2,str(grid[8,x,y,2].astype(int)),size = 7) #for plotting the number of counted cells in a gridcell
        if velplot == 1:
            velplotter(imgs[F2P],velpos[F2P,1],velpos[F2P,2],velpos[F2P,0],F2P)  
        plt.show()

plotter()