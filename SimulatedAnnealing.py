import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import skimage
from skimage.filters import gaussian, sobel
from skimage.feature.register_translation import _upsampled_dft
from numpy.fft import fft2,ifft2
from skimage.io import imread
import cv2
import random
from skimage.measure import profile_line
from scipy.signal import argrelmax, argrelmin
import copy
from copy import deepcopy
import infotracking
from infotracking.infotheory import conditional_entropy, entropy

class Cell():
    def __init__(self, pos, angle, length, radius, intensity):
        self.pos = pos
        self.angle = angle
        self.length = length
        self.radius = radius
        self.intensity = intensity
        self.orig_length = length # keep track of starting length of cell before optimization
        self.orig_angle = angle # keep track of starting angle of cell before optimization

    def draw_cv(self, img):
        ax = np.array([np.cos(self.angle), np.sin(self.angle)])
        p0 = self.pos + ax*self.length*0.5
        p1 = self.pos - ax*self.length*0.5
        r = int(self.radius)
        axperp = r*np.array((ax[1],-ax[0]))
        cv2.circle(img, tuple(p0.astype(np.int)), r, self.intensity, thickness=-1)
        cv2.circle(img, tuple(p1.astype(np.int)), r, self.intensity, thickness=-1)
        r0 = (p0+axperp).astype(np.int)
        r1 = (p0-axperp).astype(np.int)
        r2 = (p1-axperp).astype(np.int)
        r3 = (p1+axperp).astype(np.int)
        pts = np.array([r0,r1,r2,r3])
        cv2.fillConvexPoly(img, pts, self.intensity)

def draw_cells_cv(cells, w, h):
    img = np.zeros(shape=(w,h)).astype(np.uint8)
    for cell in cells:
        cell.draw_cv(img)
    return img

def model(w, h, cells): 
    im = np.array(draw_cells_cv(cells, w, h)).astype(np.float32) 
    im = im/im.max()
    im = gaussian(im, 4.)
    return im

def error_mse(data, cells):
    # Calculate error between data and test image
    # Intensity mean squared error
    w,h = data.shape
    test = model(w, h, cells)
    err = test - data
    msqerr = np.sqrt(np.mean(err*err))
    # Edges mean squared error
    edge_data = sobel(data)
    edge_test = sobel(test)
    edge_err = edge_test/edge_test.max() - edge_data/edge_data.max()
    edge_msqerr = np.mean(edge_err*edge_err)
    # Penalise unlikely changes in length and angle
    lenerr = 0
    angerr = 0
    enderr = 0
    for cell in cells:
        sigma_len = 1.
        sigma_ang = 0.1
        lenerr += (cell.length-cell.orig_length)**2 / (2.*sigma_len**2)
        angerr += (cell.angle-cell.orig_angle)**2 / (2.*sigma_ang**2)
        profile = cell_profile(data, cell)
        mx_profile = profile.max()
        enderr += max(profile[-1], profile[0])/mx_profile

    edge_weight = 0.
    len_weight = 1e-4
    ang_weight = 0. #1e-4
    end_weight = 1e-3
    #print('msqerr = %f, lenerr = %f, angerr = %f'%(msqerr,len_weight*lenerr,ang_weight*angerr))
    return msqerr + edge_weight*edge_msqerr + len_weight*lenerr + ang_weight*angerr + end_weight*enderr

def error_entropy(data, cells):
    w,h = data.shape
    test = model(w, h, cells)
    hgram, xedges, yedges = np.histogram2d( data.ravel(), test.ravel(), bins=32)
    cH = conditional_entropy(hgram, ax=0)
    H = entropy(hgram, ax=1)
    return cH/H

def fit_func(params, data, ncells):
    cells = []
    for i in range(ncells):
        pos = np.array([params[i], params[i+ncells]])
        ang = params[i+ncells*2]
        length = params[i+ncells*3]
        rad = params[i+ncells*4]
        cells.append(Cell(pos,ang,length,rad,128))
    return(error_mse(data,cells))

def minimizer(cells, data):
    pos = []
    pos = [cell.pos for cell in cells]
    posx = [p[0] for p in pos]
    posy = [p[1] for p in pos]
    ang = [cell.angle for cell in cells]
    length = [cell.length for cell in cells]
    rad = [cell.radius for cell in cells]
    params = posx + posy + ang + length + rad
    m = minimize(fit_func, params, args=(data,len(cells)), method='BFGS')
    params = m
    print('Minimized solution: ', m)
    mincells = []
    ncells = len(cells)
    for i in range(ncells):
        pos = np.array([params[i], params[i+ncells]])
        ang = params[i+ncells*2]
        length = params[i+ncells*3]
        rad = params[i+ncells*4]
        mincells.append(Cell(pos,ang,length,rad,128))
    plot_solution(mincells, data)
    print(len(mincells))
    for cell in mincells:
        print("pos = ", cell.pos, \
        ", ang = ", cell.angle, \
        ", len = ", cell.length, \
        ", rad = ", cell.radius, \
        ", intensity = ", cell.intensity)
    return(mincells, error_mse(data,mincells))




def simulated_anneal(cells, \
                        data, \
                        nt=100000, temp_scale=1e-4, \
                        dpos = 16., \
                        dang = .2, \
                        dlen = 5., \
                        drad = 1., \
                        dintensity = 10., \
                        minlen = 10., \
                        maxlen = 100., \
                        minrad = 2., \
                        maxrad = 5. \
                        ):
    # Image dimensions
    w,h = data.shape
    ncells = len(cells)

    # Initial error
    mincells = deepcopy(cells)
    bestcells = deepcopy(cells)
    minerr = error_mse(data, mincells) 
    besterr = minerr

    # Print out the starting configuration
    print('Starting configuration:')
    for cell in mincells:
        print("pos = ", cell.pos, \
          ", ang = ", cell.angle, \
          ", len = ", cell.length, \
            ", rad = ", cell.radius, \
                 ", intensity = ", cell.intensity)

    for t in range(nt):
        # Get the latest solution
        if t%100==0:
            testcells = deepcopy(bestcells)
            err = besterr
        else:
            testcells = deepcopy(mincells)
        cidx = random.randint(0,ncells-1)
        # Perturb shape parameters by random variables
        for cell in testcells[cidx:cidx+1]:
            q = random.randint(0,3)
            if q==0:
                cell.pos += [(random.random()-.5)*dpos, (random.random()-.5)*dpos]
            elif q==1:
                cell.angle += (random.random()-.5)*dang
            elif q==2:
                cell.length += (random.random()-.5)*dlen
                cell.length = np.clip(cell.length, minlen, maxlen)
            #elif q==3:
            #    cell.radius += (random.random()-.5)*drad
            #    cell.radius = np.clip(cell.radius, minrad, maxrad)
            elif q==3:
                cell.intensity += (random.random()-0.5)*dintensity
                cell.intensity = np.clip(cell.intensity,0,255)
            

        # Calculate MSE error
        err = error_mse(data, testcells) 
        
        # Calculate probability to accept change
        T = ( .75 - (t/(nt-1)) ) * temp_scale
        T = max(0.,T)
        #T = np.exp( -t/nt * 20. ) * temp_scale
        if err<minerr:
            p = 1.
        else:
            p = np.exp((minerr-err)/T)
        
        # Accept change with probability p
        if random.random()<p:
            minerr = err
            # Track best solution so far
            if minerr<besterr:
                besterr = minerr
                bestcells = deepcopy(testcells)
            mincells = deepcopy(testcells)
            '''
            print('---')
            print('Accepted move at iteration %d, probability: %f'%(t,p))
            print('Lowest error so far: %f'%besterr)
            print('Temperature: %f'%T)
            for cell in mincells:
                print("pos = ", cell.pos, \
                  ", ang = ", cell.angle, \
                  ", len = ", cell.length, \
                    ", rad = ", cell.radius, \
                         ", intensity = ", cell.intensity)
            print(err)
            '''
            
        if minerr<0.01:
            break
        # End loop

    # Reset starting length and angle of cells
    for cell in bestcells:
        cell.orig_length = cell.length
        cell.orig_angle = cell.angle

    # Return the global minimum estimate
    return bestcells, besterr

def plot_solution(mincells, data):
    plt.subplot(1,3,1)
    plt.cla()
    w,h = data.shape
    test = model(w, h, mincells)
    #test = test / test.max()
    plt.imshow(test)
    # Plot cell axes
    for cell in mincells:
        ang = cell.angle
        pos = cell.pos
        length = cell.length
        ax = np.array([np.cos(ang), np.sin(ang)])
        p0 = pos + ax*length*0.5
        p1 = pos - ax*length*0.5
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]])

    #plt.colorbar()
    plt.subplot(1,3,2)
    plt.cla()
    plt.imshow(data)
    #plt.colorbar()
    # Plot cell axes
    for cell in mincells:
        ang = cell.angle
        pos = cell.pos
        length = cell.length
        ax = np.array([np.cos(ang), np.sin(ang)])
        p0 = pos + ax*length*0.5
        p1 = pos - ax*length*0.5
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]])

def cell_profile(image, cell):
    ax = np.array([np.cos(cell.angle), np.sin(cell.angle)])
    p0 = cell.pos - ax*(cell.length*0.5 + cell.radius)
    p1 = cell.pos + ax*(cell.length*0.5 + cell.radius)
    profile = profile_line(image, (p0[1], p0[0]),(p1[1],p1[0]), order=2)
    return profile


def split_cells(im, cells, minlen):
    #print('--- Split cells ---')
    newcells = []
    plt.subplot(1,3,3)
    plt.cla()
    for cell in cells:
        profile = cell_profile(im, cell)
        plt.plot(profile, '.-')

        # Find minima of profile
        minima = argrelmin(profile)[0]
        #print('Minima ',minima)
        if len(minima)>0:
            # Calculate relative depth of minima from maximum peak
            depth = (profile.max() - profile[minima]) / profile.max()
            #print('Depth ',depth)
            
            # Find index of deepest minimum
            didx = np.argmax(depth)
            idx = minima[didx]
            #print('Indices ', idx)

            # Conditions to accept division
            cond1 = depth[didx]>0.2
            cond2 = abs(idx-len(profile)/2)/len(profile)<0.25

            ratio = float(idx)/float(len(profile)-1)
            length1 = (cell.length + cell.radius*2.) * ratio  - 2.*cell.radius
            length2 = (cell.length + cell.radius*2.) * (1.-ratio) - 2.*cell.radius
            cond3 = length1>minlen
            cond4 = length2>minlen
            if cond1 and cond2 and cond3 and cond4:
                #print("** Dividing cell at position = %d **"%idx)
                plt.plot([idx,idx],[0,1],'r--')
                ax = np.array([np.cos(cell.angle), np.sin(cell.angle)])
                p0 = cell.pos - ax*cell.length*0.5
                p1 = cell.pos + ax*cell.length*0.5
                pos1 = p0 + ax*length1*0.5
                pos2 = p1 - ax*length2*0.5 
                cell1 = Cell(pos1, cell.angle, length1, cell.radius, cell.intensity)
                cell2 = Cell(pos2, cell.angle, length2, cell.radius, cell.intensity)
                newcells.append(cell1)
                newcells.append(cell2)
            else:
                newcells.append(cell)
        else:
            newcells.append(cell)

    plt.ylim([0.,1.])
    plot_solution(newcells,im)

    # Print out the starting configuration
    print('Final configuration:')
    for cell in newcells:
        print("pos = ", cell.pos, \
          ", ang = ", cell.angle, \
          ", len = ", cell.length, \
            ", rad = ", cell.radius, \
                 ", intensity = ", cell.intensity)
        print('length change = %f, angle change = %f'%(cell.length-cell.orig_length,cell.angle-cell.orig_angle))

    return newcells

if __name__=='__main__':
    import sys
    fname = sys.argv[1]
    nframes = int(sys.argv[2])
    startframe = int(sys.argv[3])
    print(fname, nframes)
    dataall = imread(fname, plugin='tifffile')

    # Starting parameters, initial guess
    ncells = 1  
    minpos = [52.*2,36.*2]
    minang = 1.47
    minlen = 30.*2
    minrad = 2.5*2
    minintensity = 128
    cells = []
    for i in range(ncells):
        cells.append(Cell(np.array(minpos), minang, minlen, minrad, minintensity))

    plt.figure(figsize=(12,4))
    for f in range(startframe, startframe+nframes, 1):
        data = dataall[f,:,:]
        w,h = data.shape
        # Upsample image by 2x
        fim = fft2(data)
        data = np.real(_upsampled_dft(fim, (w*2,h*2), upsample_factor=2)[::-1,::-1])
        #data = (data-data.min())/(data.max()-data.min())
        data = data/data.max() 
        print("max data = ", data.max())

        minerr = 1e12
        for i in range(1):
            cells,err = simulated_anneal(cells, data, nt = 200*len(cells)**2)
            #cells,err = minimizer(cells, data) 
            cells = split_cells(data, cells, minlen=10.)
            if err<minerr:
                mincells = deepcopy(cells)
        plot_solution(mincells, data)
        plt.savefig('simulated_annealing_frame%04d.png'%f)
        plt.pause(0.1)

    print('*** DONE ***')
    plt.pause(0)
