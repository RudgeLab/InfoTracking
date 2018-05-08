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

def velplotter(image,position1,position2,velocity,j):
    fig1 = plt.figure()
    plt.imshow(image)
    plt.plot(position1[:,0],position1[:,1],"ro",markersize = 0.5)
    plt.quiver(position1[:,0],position1[:,1],velocity[:,0],velocity[:,1])
    #plt.show()
    #fig1.savefig('velstep-00'+str(startframe+j)+'0.pdf')

'''-------MAIN-----'''
plt.ioff()

dt=1
nframes = 10
startframe = 50
 
#fname = "/Users/timrudge/cellmodeller/data/weiner-17-12-07-17-56/step-%04d0.pickle"
fname = "/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-%04d0.pickle"
#fname2 = "/Users/timrudge/cellmodeller/data/weiner-17-12-07-17-56/step-%04d0.png"
fname2 = "/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-%04d0.png"

data = np.array([cPickle.load(open(fname%(startframe+i))) for i in range(nframes)])
imgs = np.array([plt.imread(open(fname2%(startframe+i))) for i in range(nframes)])


csa = np.array([element['cellStates'] for element in data])


worldsize = 250.0
imagesize = imgs[0].shape[0]

resizing= imagesize/worldsize 
C = imagesize/2 


print "worldsize = ", worldsize
print "imagesize = ", imagesize
print "resizing factor = ", resizing


velpos = np.array([Pickledx(csa[i],csa[i+1],dt,resizing) for i in range(nframes-1)]) #dim (nframes-1,3) with dx,x1,x2

#pos1 = np.array([[-a,b,c] for (a,b,c) in pos1]) #fix axis, after noticing how to fix: a simpler way would be pos[:,0] = -pos[:,0]

for element in velpos:
    element[1] = np.array([[C+a,C-b,c] for (a,b,c) in element[1]])
for item in velpos:
    item[2] = np.array([[C+a,C-b,c] for (a,b,c) in item[2]])
print("Done calculating velocities")
'''NOT NEEDED
for i in range(len(velpos)):
    velplotter(imgs[i],velpos[i,1],velpos[i,2],velpos[i,0],i)
'''


def sumtogrid(posit,vels,grid,gxs,gys):
    x,y = posit[0],posit[1]
    xchk,ychk = x/gxs, y/gys
    grid[xchk,ychk] += np.array([vels[0],vels[1]])
    return grid
    
def plotgrid():
    for i in range(gx):
        plt.hlines(i*dgx,0,imagesize)
    for j in range(gy):
        plt.vlines(j*dgy,0,imagesize)
        
gridfac = 64
gx,gy = int(np.floor(imagesize/gridfac)),int(np.floor(imagesize/gridfac))
print "Grid dimensions: ",gx,gy
dgx,dgy = gridfac,gridfac
pos = np.zeros((nframes-1,gx,gy,2))
for step in range(nframes-1):
    poscheck1 = (velpos[step,1]).astype(int)
    vel2add = velpos[step,0]
    for i in range(len(poscheck1)):
        pos[step] = sumtogrid(poscheck1[i],vel2add[i],pos[step],dgx,dgy)
#pos is now a grid of nframes,gx,gy,sumofvelocities
        
#plots:
        
for x in range(len(pos[8])): #8th as example
    for y in range(len(pos[8])):
        plt.quiver((dgx*x)+gridfac/2,(dgy*y)+gridfac/2,pos[1,x,y,0],pos[1,x,y,1])
plt.imshow(imgs[8])
plotgrid()
plt.show()