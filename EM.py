import numpy as np
from numpy.fft import fft2,ifft2,fftshift
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.signal import correlate2d
import math
import infotheory

plotting = False
save_images = False



def analyse_region(im1,im2, w,h, vx_max, vy_max, px1,py1, px2,py2, nbins, range_mx):
    '''
    Compute mutual information between regions in image pairs over a range of
    offsets. This gives an estimate of the structural similarity between the
    image pair in the specified region. 

    im1,im2 = image pair, numpy float arrays with same shape

    w,h = dimensions of image region are 2w+1, 2h+1

    vx_max,vy_max = tuple giving the maximum offset in each image dimension, in each
    direction

    px,py = position of centre in im1
    px2,py2 = position of centre in im2

    nbins = number of bins to use for image histograms

    returns: float array at offsets in [-vx_max:vx_max,
    -vy_max,vy_max] of:
            hy = entropy of im2 = H(im2)
            hy_cond_x = H(im2|im1)
            mi = mutual information = I(im1,im2)
    '''


    vw = vx_max*2 + 1
    vh = vy_max*2 + 1

    # Output arrays
    hy = np.zeros((vw,vh))
    hy_cond_x = np.zeros((vw,vh))
    mi = np.zeros((vw,vh))
    hz = np.zeros((vw,vh))

    im1_roi = im1[px1:px1+w, py1:py1+h]
    hgram, edges = np.histogram( im1_roi.ravel(), \
            bins=nbins, \
            range=(0,range_mx))
    Hx = infotheory.entropy(hgram)
    for vx in range(-vx_max,vx_max+1):
        for vy in range(-vx_max,vy_max+1):
            #print vx, vy
            #im2_roi_offset = im2[px2+vx:px2+w+vx, py2+vy:py2+h+vy]
            #im2_roi = im2[px2:px2+w, py2:py2+h]
            im1_roi_offset = im1[px2+vx:px2+vx+w, py2+vy:py2+vy+h]
            im2_roi_offset = im2[px2+vx:px2+vx+w, py2+vy:py2+vy+h]
            im2_roi = im2[px2:px2+w, py2:py2+h]
            '''
            print "vx ", vx
            print "vy ", vy
            print "px2 ", px2
            print "py2 ", py2
            print "w,h ", w,h
            print im2_roi_offset
            print im2_roi
            '''

            hgram_diff, edges = np.histogram( (im2_roi_offset-im1_roi).ravel(), \
                                                    bins=nbins, \
                                                    range=(0,range_mx))

            hgram_offset_self, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im1_roi_offset.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,range_mx),(0,range_mx)])
            hgram_offset, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im2_roi_offset.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,range_mx),(0,range_mx)])
            hy_val_offset = infotheory.entropy(hgram_offset, ax=0)
            hgram, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im2_roi.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,range_mx),(0,range_mx)])
            hy_val = infotheory.entropy(hgram, ax=0)
            if hy_val<1.0:
                #print 'Entropy of region (%d,%d) too low (%f), skipping'%(px1,py1,hy_val)
                return None

            hy_cond_x_val  = infotheory.conditional_entropy(hgram, ax=1)
            hy_cond_x_val_offset  = infotheory.conditional_entropy(hgram_offset, ax=1)
            hy_cond_x_val_offset_self  = infotheory.conditional_entropy(hgram_offset_self, ax=1)
            mi_val = infotheory.mutual_information(hgram)
            mi_val_offset = infotheory.mutual_information(hgram_offset)

            hy[vx+vx_max,vy+vy_max] = hy_val_offset
            hy_cond_x[vx+vx_max,vy+vy_max] = hy_cond_x_val_offset
            mi[vx+vx_max,vy+vy_max] = mi_val_offset

            hz[vx+vx_max,vy+vy_max] = infotheory.entropy(hgram_diff) + hy_val_offset
            #hy_cond_x_val_offset_self - hy_cond_x_val_offset 
            #mi_val - infotheory.joint_entropy(hgram) + infotheory.joint_entropy(hgram_offset)
    # Return array grid of entropy measures, and 
    #plt.figure()
    #plt.plot(-hz[9,:])
    return hy,hy_cond_x, mi, hz

def velocity_dist(L, vx_max,vy_max):
    '''
    Estimate velocity variance by maximising expectation of log likelihood
    
    Computes the weighted sums over the velocity range: 
        
        mean = z0 = (1/N)*sum(L*z)
        var = (1/N)*sum{ L*(z-z0)^2 }

    z = displacement
    N = total number of displacement grid points
    
    L = log likelihood array
    
    vx_max, vy_max = range of displacement grid points
    '''
    
    N = len(L.ravel())
    x = np.arange(-vx_max,vx_max+1)
    y = np.arange(-vy_max,vy_max+1)
    vy,vx = np.meshgrid(x,y)
    mean = np.sum(v*L)
    var = np.sum(v*v*L)/N
    return var
    
def find_peak(arr):
    '''
    Find location (x,y) of peak in values of array arr

    returns: (x,y) position of peak
    '''
    w,h = arr.shape
    ww = (w-1)/2
    hh = (h-1)/2

    mx = np.max(arr,axis=(0,1))
    mn = np.min(arr,axis=(0,1))
    pk = np.abs(arr-mn) > 0.95*(mx-mn)
    pky,pkx = np.meshgrid(np.arange(-ww,ww+1), np.arange(-hh,hh+1))
    x = np.sum(pkx*pk)/np.sum(pk)
    y = np.sum(pky*pk)/np.sum(pk)
    return x,y

# Compute conditional entropy at different uniform velocities
def track_cond_entropy(im1,im2, vx_max,vy_max, px1,py1, gs, nbins=256, range_mx=2**8, ofname=None):
    # Compute mutual information between image regions over offset grid
    region_analysis = analyse_region(im1,im2, \
                                                            gs,gs, \
                                                            vx_max,vy_max, \
                                                            px1,py1, \
                                                            px1,py1, \
                                                            nbins, range_mx)
    if region_analysis:
        hy_grid,hy_cond_x_grid,mi_grid,hz_grid = region_analysis
    else:
        return None

    # Original ROIs
    im1_roi = im1[px1:px1+gs,py1:py1+gs]
    im2_roi = im2[px1:px1+gs,py1:py1+gs]

    if np.max(hy_grid, axis=(0,1))<1.0:
        # Entropy too low, probably background
        px2,py2 = px1,py1
        return px2,py2,0,0,im2_roi
    else:
        #Find peak in I(im1,im2)/H(im2) to compute mean velocity estimate
        ll = -hy_cond_x_grid  #mi_grid/hy_grid 
        zxpk,zypk = find_peak(ll)

        # Estimate new mean position as peak of I(im2,im2)/H(im2)
        px2 = px1 + zxpk
        py2 = py1 + zypk
            
        # Shifted 2nd image ROI
        im2_roi_shifted = im2[px2:px2+gs,py2:py2+gs]

        return px2,py2,ll,hz_grid,im2_roi_shifted




def main(fname,startframe,nframes,step,gridfact, forwards = True, GridMethod = None):
    '''
    Input args:

    filename_pattern, start_frame, number_frames
    '''
    '''
    import sys
    if plotting:
        plt.ion()
        plt.figure(figsize=(12,4))
    '''

    # Load images
    '''
    fname = sys.argv[1]

    startframe = int(sys.argv[2])
    nframes = int(sys.argv[3])
    step = int(sys.argv[4])
    '''
    if forwards == True:
        im1 = [plt.imread(fname%(startframe+(i)*step)).astype(np.float32) for i in range(nframes)]
        im2 = [plt.imread(fname%(startframe+(i+1)*step)).astype(np.float32) for i in range(nframes)]
    elif forwards == False:
        im1 = [plt.imread(fname%(startframe+(nframes-i)*step)).astype(np.float32) for i in range(nframes)] #backwards
        im2 = [plt.imread(fname%(startframe+(nframes-1-i)*step)).astype(np.float32) for i in range(nframes)] #backwards
        
    w,h,c = im1[0].shape[0], im1[0].shape[1], 1
    

    # Grid dimensions and spacing for regions of interest
    gx,gy = int(np.floor(w/gridfact)),int(np.floor(h/gridfact))
    dgx,dgy = gridfact, gridfact
    gside = gridfact

    print "Image dimensions: ",w,h,c
    print "Grid dimensions: ",gx,gy

    print "Image intensity range:"
    mx1 = np.max(np.array(im1))
    mn1 = np.min(np.array(im1))
    mx2 = np.max(np.array(im2))
    mn2 = np.min(np.array(im2))
    print mn1, mx1
    print mn2, mx2

    if GridMethod == None:
        GridMethod = input("Select grid method (1) for tracking grid or (2) for reset grid: ") 

    # Filter images to remove noise
    from scipy.ndimage.filters import gaussian_filter
    im1 = [gaussian_filter(im1[i],1) for i in range(nframes)]
    im2 = [gaussian_filter(im2[i],1) for i in range(nframes)]


    # Compute velocity and position of ROIs based on maximum mutual information translation
    vmax = 25 
    pos = np.zeros((gx,gy,nframes,2))
    llikelihood = np.zeros((gx,gy,nframes,vmax*2+1,vmax*2+1))
    roi = np.zeros((gx,gy,nframes,gside,gside))
    grid = np.zeros((nframes,gx,gy,5))

    # Set initial grid positions
    px0 = 0 #gside + vmax
    py0 = 0 #gside + vmax
    if GridMethod == 1:
        for ix in range(gx):
            for iy in range(gy):
                pos[ix,iy,0,:] = [px0+ix*dgx,py0+iy*dgy]
                grid[0,ix,iy,:] = [0,0,1,px0+ix*dgx,py0+iy*dgy]
    
        for i in range(nframes-1):
            print '------------ Step %d ---------'%i
            print ' Image pair:'
            print '  ', fname%(startframe+(nframes-i)*step)
            print '  ', fname%(startframe+(nframes-1-i)*step)
            for ix in range(gx):
                for iy in range(gy):
                    ofname = 'gridtesting/im2-pos%d_%d_step%04d.tif'%(pos[ix,iy,0,0],pos[ix,iy,0,1],i)
                    px = int(pos[ix,iy,i,0])
                    py = int(pos[ix,iy,i,1])
                    #print px,py
                    if px<vmax or px>=w-gside-vmax or py<vmax or py>=h-gside-vmax: 
                        #print "Grid square outside image, skipping"
                        tracking = None
                    else:
                        tracking = track_cond_entropy(im1[i], im2[i], \
                                                vmax, vmax, \
                                                px,py, \
                                                gside, \
                                                nbins=16, range_mx=max(mx1,mx2), \
                                                ofname=ofname)
                    grid[i+1,ix,iy] = [0,0,0,grid[i,ix,iy,3],grid[i,ix,iy,4]]
                    if tracking:
                        px2,py2,ll,hzgrid,im2_roi = tracking
                        pos[ix,iy,i+1,:] = [px2,py2]
                        print "vel ", px2-px, py2-py
                        llikelihood[ix,iy,i+1,:,:] = ll
                        #plt.figure()
                        #plt.plot(hzgrid.ravel(), ll.ravel())
                        roi[ix,iy,i+1,:,:] = im2_roi
                        grid[i,ix,iy,:] += [px2-px,py2-py,1,0,0]
                        grid[i+1,ix,iy] += [0,0,0,grid[i,ix,iy,0],grid[i,ix,iy,1]]
                        '''print 'vel = ', px2-px, py2-py'''
                    else:
                        pos[ix,iy,i+1,:] = [px,py]
                        llikelihood[ix,iy,i+1,:,:] = 0.0
                        roi[ix,iy,i+1,:,:] = 0
                        grid[i,ix,iy,:] = [0,0,0,px,py]
    
                    #plt.ylim([0,30])
                    #if i==0:
                    #    plt.colorbar()
    
            #print 'pos[i+1] =', pos[ix,iy,i+1,:]
        
        pos.tofile('pos.np', sep=',')
        roi.tofile('roi.np', sep=',')
        llikelihood.tofile('ll.np', sep=',')
        
        print "Done"
        return grid

# Run analysis
if __name__ == "__main__": 
    main('/home/timrudge/cellmodeller/data/info_tracking-18-06-08-12-59/step-%05d.png' \
            ,360 \
            ,5 \
            ,2 \
            ,64 \
            ,False, GridMethod = 1)
    pass
