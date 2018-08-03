import numpy as np

class EnsembleState:
    def __init__(self, pos, vel, w, h):
        # Initialise the ensemble at a given location
        self.w = w
        self.h = h
        self.pos = pos
        self.vel = vel

class ImageEnsembleState():
    def __init__(self, pos, w, h, image_roi1, image_roi2):
        # Ensemble computed from image pair, stores regions of interest at t and t+1
        self.w = w
        self.h = h
        self.pos = pos
        self.vel = vel
        self.image1 = image1
        self.image2 = image2

    def entropy_map(self, vx_max=7, vy_max=7, nbins=16, hmin=1.0):
        # Compute conditional entropy H(I2|I1) at each displacement
        vw = vx_max*2 + 1
        vh = vy_max*2 + 1
        hy_cond_x = np.zeros((vw,vh))
        range_mx1 = np.max(image_roi1)
        range_mx2 = np.max(image_roi2)
        range_mx = max(range_mx1, range_mx2)

        # Entropy of image regions, if too low cannot compute velocity
        hgram, edges = np.histogram( self.image_roi1.ravel(), \
            bins=nbins, \
            range=(0,range_mx))
        hx = infotheory.entropy(hgram)
        if hx>hmin:
            im1_roi = im1[pos[0]:pos[0]+w, pos[1]:pos[1]+h]
            for vx in range(-vx_max,vx_max+1):
                for vy in range(-vx_max,vy_max+1):
                    im2_roi = im2[pos[0]+vx:pos[0]+w+vx, pos[1]+vy:pos[1]+h+vy]
                    hgram_offset, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                        im2_roi.ravel(), \
                                                        bins=nbins, \
                                                        range=[(0,range_mx),(0,range_mx)])
                    hy_cond_x_val  = infotheory.conditional_entropy(hgram_offset, ax=1)
                    hy_cond_x[vx+vx_max,vy+vy_max] = hy_cond_x_val
        # Return the map of conditional entropy at each offset
        return hy_cond_x

    def compute_mean_velocity(self):

    def 

class Ensemble:
    # Set of ensemble states at each time t
    def __init__(self, init_state):
        self.states = {0: init_state}
