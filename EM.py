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



def analyse_region(im1,im2, w,h, vx_max, vy_max, px1,py1, px2,py2, nbins):
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

    im1_roi = im1[px1-w:px1+w+1, py1-h:py1+h+1]
    for vx in range(-vx_max,vx_max+1):
        for vy in range(-vx_max,vy_max+1):
            #print vx, vy
            im2_roi_offset = im2[px2-w+vx:px2+w+vx+1, py2-h+vy:py2+h+vy+1]
            im2_roi = im2[px2-w:px2+w+1, py2-h:py2+h+1]
            hgram_offset, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im2_roi_offset.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,2**16),(0,2**16)])
            hy_val_offset = infotheory.entropy(hgram_offset, ax=0)
            hgram, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im2_roi.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,2**16),(0,2**16)])
            hy_val = infotheory.entropy(hgram, ax=0)
            if hy_val<1.0:
                print 'Entropy of region (%d,%d) too low (%f), skipping'%(px1,py1,hy_val)
                return None
            hy[vx+vx_max,vy+vy_max] = hy_val_offset
            hy_cond_x_val  = infotheory.conditional_entropy(hgram_offset, ax=1)
            mi_val = infotheory.mutual_information(hgram_offset)

            hy[vx+vx_max,vy+vy_max] = hy_val
            hy_cond_x[vx+vx_max,vy+vy_max] = hy_cond_x_val
            mi[vx+vx_max,vy+vy_max] = mi_val

            hz[vx+vx_max,vy+vy_max] = mi_val - infotheory.joint_entropy(hgram) + infotheory.joint_entropy(hgram_offset)
    # Return array grid of entropy measures, and 
    return hy,hy_cond_x, mi, hz

def find_peak(arr):
    '''
    Find location (x,y) of peak in values of array arr

    returns: (x,y) position of peak
    '''
    w,h = arr.shape
    ww = (w-1)/2
    hh = (h-1)/2

    pk = arr>0.99*np.max(arr,axis=(0,1))
    pky,pkx = np.meshgrid(np.arange(-ww,ww+1), np.arange(-hh,hh+1))
    x = np.sum(pkx*pk)/np.sum(pk)
    y = np.sum(pky*pk)/np.sum(pk)
    return x,y

# Compute conditional entropy at different uniform velocities
def track_cond_entropy(im1,im2, vx_max,vy_max, px1,py1, gs, nbins=256, ofname=None):
    # Compute mutual information between image regions over offset grid
    region_analysis = analyse_region(im1,im2, \
                                                            gs,gs, \
                                                            vx_max,vy_max, \
                                                            px1,py1, \
                                                            px1,py1, \
                                                            nbins)
    if region_analysis:
        hy_grid,hy_cond_x_grid,mi_grid,hz_grid = region_analysis
    else:
        return None

    # Original ROIs
    im1_roi = im1[px1-gs:px1+gs,py1-gs:py1+gs]
    im2_roi = im2[px1-gs:px1+gs,py1-gs:py1+gs]

    if np.max(hy_grid, axis=(0,1))<1.0:
        # Entropy too low, probably background
        px2,py2 = px1,py1
        return px2,py2,0,im2_roi
    else:
        #Find peak in I(im1,im2)/H(im2) to compute mean velocity estimate
        ll = mi_grid/hy_grid
        zxpk,zypk = find_peak(ll)

        # Estimate new mean position as peak of I(im2,im2)/H(im2)
        px2 = px1 + zxpk
        py2 = py1 + zypk
            
        # Shifted 2nd image ROI
        im2_roi_shifted = im2[px2-gs:px2+gs,py2-gs:py2+gs]

        return px2,py2,ll,im2_roi_shifted




def main():
    '''
    Input args:

    filename_base, filename_extension, start_frame, number_frames
    '''

    import sys
    if plotting:
        plt.ion()
        plt.figure(figsize=(12,4))


    # Load images

    fnamebase = sys.argv[1]
    fname = fnamebase + sys.argv[2]

    startframe = int(sys.argv[3])
    nframes = int(sys.argv[4])
    im1 = [plt.imread(fname%(startframe+i*2)).astype(np.float32) for i in range(nframes)]
    im2 = [plt.imread(fname%(startframe+2+i*2)).astype(np.float32) for i in range(nframes)]

    w,h = im1[0].shape
    w,h = im1[0].shape

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


    # Compute velocity and position of ROIs based on maximum mutual information translation
    vmax = 7
    pos = np.zeros((gx,gy,nframes,2))
    llikelihood = np.zeros((gx,gy,nframes,vmax*2+1,vmax*2+1))
    roi = np.zeros((gx,gy,nframes,gside,gside))


    # Set initial grid positions
    px0 = gside + vmax
    py0 = gside + vmax
    for ix in range(gx):
        for iy in range(gy):
            pos[ix,iy,0,:] = [px0+ix*dgx,py0+iy*dgy]
    for i in range(nframes-1):
        print '------------ Step %d ---------'%i
        for ix in range(gx):
            for iy in range(gy):
                ofname = 'gridtesting/im2-pos%d_%d_step%04d.tif'%(pos[ix,iy,0,0],pos[ix,iy,0,1],i)
                px = int(pos[ix,iy,i,0])
                py = int(pos[ix,iy,i,1])
                print px,py
                if px<gside+vmax or px>=w-gside-vmax or py<gside+vmax or py>=h-gside-vmax: 
                    print "Grid square outside image, skipping"
                    tracking = None
                else:
                    tracking = track_cond_entropy(im1[i], im2[i], \
                                            vmax, vmax, \
                                            px,py, \
                                            gside/2.0, \
                                            nbins=256, ofname=ofname)
                if tracking:
                    px2,py2,ll,im2_roi = tracking
                    pos[ix,iy,i+1,:] = [px2,py2]
                    llikelihood[ix,iy,i+1,:,:] = ll
                    roi[ix,iy,i+1,:,:] = im2_roi
                    print 'vel = ', px2-px, py2-py
                else:
                    pos[ix,iy,i+1,:] = [px,py]
                    llikelihood[ix,iy,i+1,:,:] = 0.0
                    roi[ix,iy,i+1,:,:] = 0

                #plt.ylim([0,30])
                #if i==0:
                #    plt.colorbar()

        #print 'pos[i+1] =', pos[ix,iy,i+1,:]

    pos.tofile('pos.np', sep=',')
    roi.tofile('roi.np', sep=',')
    llikelihood.tofile('ll.np', sep=',')


# Run analysis
if __name__ == "__main__": 
    main()
