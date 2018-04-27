import cPickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def pos2pixel(ps,factr):
    return factr*ps

def Pickledx(cs1,cs2,dt,fact): #cellstate1,cellstate2,dt,scalation factor
    dx,arraypos1,arraypos2 = [],[],[]
    i = 0
    ids1 = np.array([cell1.id for (id,cell1) in cs1.iteritems()])
    ids2 = np.array([cell.id for (id,cell) in cs2.iteritems()])
    for (id,cell) in cs1.iteritems():
        try:
            pos1,pos2 = np.array(cell.pos), np.array(cs2[id].pos)
            dx.append(pos2pixel(pos2-pos1,fact))
            arraypos1.append(pos1)
            arraypos2.append(pos2)
        except KeyError:
            print 'Data2 has no id',id
            i+=1
    dx,arraypos1,arraypos2 = np.array(dx),np.array(arraypos1),np.array(arraypos2)
    print 'Skipped', i,'ids, which should match this number:', len(ids2)-len(ids1)
    return dx/dt, arraypos1, arraypos2

def velplotter(image,position1,position2,velocity):
    plt.figure()
    plt.imshow(image)
    plt.plot(position1[:,0],position1[:,1],"ro",markersize = 1)
    plt.quiver(position1[:,0],position1[:,1],velocity[:,0],velocity[:,1])
    plt.show(block=False)
    
data1 = cPickle.load(open("/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-00520.pickle", 'rb'))
data2 = cPickle.load(open("/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-00530.pickle", 'rb'))
img = mpimg.imread("/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-00520.png")

csa = data1['cellStates']
csb = data2['cellStates']

worldsize = 250
imagesize = 1181

#resizing= imagesize/worldsize #not so good
#C = imagesize/2 #not so good

C = 590.5
resizing = 4.724

vel,pos1,pos2 = Pickledx(csa,csb,1,resizing)
pos1 = np.array([[-a,b,c] for (a,b,c) in pos1]) #fix axis, after noticing how to fix: a simpler way would be pos[:,0] = -pos[:,0]
pos2 = np.array([[-a,b,c] for (a,b,c) in pos2]) 
pos1,pos2 = C - pos2pixel(pos1,resizing), C - pos2pixel(pos2,resizing) #pickle's 0 is in middle of colony

velplotter(img,pos1,pos2,vel)
'''
# Grid dimensions and spacing for regions of interest
    gx,gy = int(np.floor(w/64))-1,int(np.floor(h/64))-1
    dgx,dgy = 64,64
    gside = 64

    print "Image dimensions: ",w,h
    print "Grid dimensions: ",gx,gy

    print "Image intensity range:"
    print np.max(im1), np.min(im1)
    print np.max(im2), np.min(im2)


    # Filter images to remove noise
    from scipy.ndimage.filters import gaussian_filter
    im1 = [gaussian_filter(im1[i],1) for i in range(nframes)]
    im2 = [gaussian_filter(im2[i],1) for i in range(nframes)]
'''