import cPickle
import numpy as np

def pos2pixel(ps,factr):
    return factr*ps

def Pickledx(cs1,cs2,dt,fact): #cellstate1,cellstate2,dt,scalation factor
    dx = []
    i = 0
    ids1 = np.array([cell1.id for (id,cell1) in cs1.iteritems()])
    ids2 = np.array([cell.id for (id,cell) in cs2.iteritems()])
    for (id,cell) in cs1.iteritems():
        try:
            x = np.array(cell.pos)-np.array(cs2[id].pos)
            dx.append(pos2pixel(x,fact))
        except KeyError:
            print 'Data2 has no id',id
            i+=1
    dx = np.array(dx)
    print 'Skipped', i,'ids, which should match this number:', len(ids2)-len(ids1)
    return dx/dt

data1 = cPickle.load(open("/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-00450.pickle", 'rb'))
data2 = cPickle.load(open("/Users/Ignacio/cellmodeller/data/Tutorial_1a-18-04-10-17-59/step-00460.pickle", 'rb'))
csa = data1['cellStates']
csb = data2['cellStates']
worldsize = 250
imagesize = 1181

'''
#not needed:
pos1 = np.array([cell1.pos for (id,cell1) in cs1.iteritems()])
pos2 = np.array([cell2.pos for (id,cell2) in cs2.iteritems()])
'''

vel = Pickledx(csa,csb,10,imagesize/worldsize)
