import numpy as np
import skimage
from skimage.io import imsave
import infotheory

class EnsembleState:
    def __init__(self, pos, vel, w, h):
        # Initialise the ensemble at a given location
        self.w = w
        self.h = h
        self.pos = pos
        self.vel = vel

class ImageEnsembleState():
    def __init__(self, pos1, t1, t2, image1, image2, w=64, h=64):
        # Ensemble computed from image pair, stores regions of interest at t and t+1
        self.t1 = t1
        self.t2 = t2
        self.w = w
        self.h = h
        self.pos1 = pos1
        self.vel = vel
        self.image1 = image1
        self.image2 = image2
        self.im1_roi = self.image1[pos[0]:pos[0]+w, pos[1]:pos[1]+h]

    def entropy_map(self, vx_max=7, vy_max=7, nbins=16, hmin=1.0):
        # Compute conditional entropy H(I2|I1) at each displacement
        vw = vx_max*2 + 1
        vh = vy_max*2 + 1
        hy_cond_x = np.zeros((vw,vh))

        # Entropy of image region, if too low cannot compute velocity
        hgram, edges = np.histogram( self.im1_roi.ravel(), \
            bins=nbins, \
            range=(0,range_mx))
        hx = infotheory.entropy(hgram)
        if hx>hmin:
            for vx in range(-vx_max,vx_max+1):
                for vy in range(-vx_max,vy_max+1):
                    im2_roi = self.image2[self.pos1[0]+vx:self.pos1[0]+self.w+vx, \
                                                self.pos1[1]+vy:self.pos1[1]+self.h+vy]
                    hgram_offset, xedges, yedges = np.histogram2d( self.im1_roi.ravel(), \
                                                        im2_roi.ravel(), \
                                                        bins=nbins)
                    hy_cond_x_val  = infotheory.conditional_entropy(hgram_offset, ax=1)
                    hy_cond_x[vx+vx_max,vy+vy_max] = hy_cond_x_val
        # Return the map of conditional entropy at each offset
        return hy_cond_x

    def compute_mean_velocity(self, llmap, threshold=0.75):
        '''
        Compute the mean velocity of the ensemble as the offset with maximum
        log likelihood, using the grid llmap 

        Sub-pixel peak position is found from weighted sum of offsets where:
            (llmap-minimum)>threshold*(maximum-minimum)

        Returns velocity in x,y and maximum value of log likelihood
        '''
        w,h = llmap.shape
        ww = (w-1)/2
        hh = (h-1)/2

        mx = np.max(llmap,axis=(0,1))
        mn = np.min(llmap,axis=(0,1))
        pk = np.abs(llmap-mn) > threshold*(mx-mn)
        pky,pkx = np.meshgrid(np.arange(-ww,ww+1), np.arange(-hh,hh+1))
        weight = pk*(llmap-mn)
        vx = np.sum(pkx*weight)/np.sum(weight)
        vy = np.sum(pky*weight)/np.sum(weight)

        # Store velocity, image roi in next time step, and maximum log likelihood
        self.vel = [vx,vy]
        self.im2_roi = self.image2[self.pos1[0]+vx:pos1[0]+self.w+vx, \
                                    self.pos1[1]+vy:self.pos1[1]+self.h+vy]
        self.pos2 = self.pos1 + self.vel
        self.maxll = mx
        return self.vel

    def save_rois(self, file_pattern1, file_pattern2):
        fname1 = file_pattern%(self.t1)
        fname2 = file_pattern%(self.t2)
        imsave(fname1, self.im1_roi, plugin='tifffile')
        imsave(fname1, self.im2_roi, plugin='tifffile')

    def save_quiver(self, file_pattern, **kwargs):
        # Generate a quiver plot, with image background, passing kwargs to quiver call
        plt.figure()
        plt.imshow(self.image1)
        plt.quiver(


class EnsembleGrid:
    # Set of ensemble states at each time t
    def __init__(self, init_state):
        # Dictionary mapping time t to ensemble state
        self.states = {0: init_state}

    def __getitem__(self, t):
        return self.states[t]

    def __setitem__(self, t, state)
        self.states[t] = state

    def pos(self):
        # Return array of ensemble positions at each time
        pos = np.array([s.pos1 for s in self.states])
