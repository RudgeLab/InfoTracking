import numpy as np
import skimage
from skimage.io import imsave
import infotracking.infotheory as infotheory
import matplotlib
import matplotlib.pyplot as plt
import os
from skimage.feature.register_translation import _upsampled_dft
from numpy.fft import fft2

nbins = 16
hmin = 1.0

class EnsembleState():
    def __init__(self, pos1, t1, t2, image1, image2, mask, w=64, h=64):
        # Ensemble computed from image pair, stores regions of interest at t and t+1
        self.live = True
        self.t1 = t1
        self.t2 = t2
        self.w = w
        self.h = h
        self.pos1 = pos1
        self.pos2 = pos1
        self.vel = [0,0]
        self.image1 = image1
        self.image2 = image2
        self.mask = mask
        self.max_ll = 0
        # Size of images
        s = image1.shape
        self.imw = s[0] 
        self.imh = s[1]
        if len(s)==3:
            self.imd = s[2]
        else:
            self.imd = 1
        self.im1_roi = np.zeros((self.w, self.h, self.imd))
        self.im2_roi = np.zeros((self.w, self.h, self.imd))
        self.im2_roi_orig = np.zeros((self.w, self.h, self.imd))
        self.llmap = np.zeros((15,15))
        self.weight = np.zeros((15,15))


    def entropy_map(self, vx_max=7, vy_max=7, nbins=nbins, hmin=0.0):
        # Compute conditional entropy H(I2|I1) at each displacement
        vw = vx_max*2 + 1
        vh = vy_max*2 + 1
        hy_cond_x = np.zeros((vw,vh))

        # ROI in first image
        ipx1 = int(self.pos1[0])
        ipy1 = int(self.pos1[1])
        # If ROI of first image falls outside image boundary return empty map
        # and set empty ROIs
        self.im1_roi = np.zeros((self.w,self.h,self.imd))
        self.im2_roi = np.zeros((self.w,self.h,self.imd))
        if ipx1<0 or ipy1<0 or ipx1+self.w>=self.imw or ipy1+self.h>=self.imh:
            return False, hy_cond_x
        # Check mask
        self.mask_roi = self.mask[ipx1:ipx1+self.w, \
                                    ipy1:ipy1+self.h]
        if np.mean(self.mask_roi)<1:
            # Outside colony
            self.live = False
            return False, hy_cond_x

        # Analyse image region
        self.im1_roi = self.image1[ipx1:ipx1+self.w, \
                                    ipy1:ipy1+self.h]


        # Entropy of image region, if too low cannot compute velocity
        hgram, edges = np.histogram( self.im1_roi.ravel(), \
            bins=nbins)
        hx = infotheory.entropy(hgram)
        #if hx<hmin:
            # Image region has low entropy
            #print("Entropy too low")
            #return False, hy_cond_x

        for vx in range(-vx_max,vx_max+1):
            for vy in range(-vx_max,vy_max+1):
                ipx2 = int(self.pos1[0]+vx)
                ipy2 = int(self.pos1[1]+vy)
                # Check roi is inside image, otherwise entropy is zero
                if ipx2>=0 and ipy2>=0 and ipx2+self.w<self.imw and ipy2+self.h<self.imh:
                    im2_roi = self.image2[ipx2:ipx2+self.w, \
                                                ipy2:ipy2+self.h]
                    hgram_offset, xedges, yedges = np.histogram2d( self.im1_roi.ravel(), \
                                                        im2_roi.ravel(), \
                                                        bins=nbins)
                    hy_cond_x_val  = infotheory.conditional_entropy(hgram_offset, ax=1) #-infotheory.mutual_information(hgram_offset)
                    hy_cond_x[vx+vx_max,vy+vy_max] = hy_cond_x_val
        # Return the map of conditional entropy at each offset
        return True,hy_cond_x

    def find_peak(self, im):
        # Find peak in array im by upsampling and locating maximum
        fim = fft2(im)
        upim = np.real(_upsampled_dft(fim, im.shape[0]*4, 2)[::-1,::-1])
        peaky,peakx = np.where(upim==np.max(upim))
        pos = np.array([peakx[0],peaky[0]]) * 0.25 
        return pos

    def compute_mean_velocity(self, llmap, threshold=0.75):
        '''
        Compute the mean velocity of the ensemble as the offset with maximum
        log likelihood, using the grid llmap 

        Sub-pixel peak position is found from weighted sum of offsets where:
            (llmap-minimum)>threshold*(maximum-minimum)

        Returns velocity in x,y and maximum value of log likelihood
        '''
        # Size of map
        w,h = llmap.shape
        ww = (w-1)/2
        hh = (h-1)/2

        mx = np.max(llmap,axis=(0,1))
        mn = np.min(llmap,axis=(0,1))
        if mx-mn<1e-6:
            vx,vy = 0,0
            print("No minumum found")
        else:
            pky,pkx = np.meshgrid(np.arange(-ww,ww+1), np.arange(-hh,hh+1))
            #pk = llmap-mn > threshold*(mx-mn)
            #weight = pk*(llmap-mn)
            #vx = np.sum(pkx*weight)/np.sum(weight)
            #vy = np.sum(pky*weight)/np.sum(weight)
            std = ww/2.
            self.weight = np.exp(llmap*2)  * np.exp(-(pkx**2 + pky**2)/(2*std**2))
            pos = self.find_peak(self.weight)
            vx = pos[0] - ww - 1
            vy = pos[1] - hh - 1

        # Store velocity, image roi in next time step, and maximum log likelihood
        self.vel = [-vx,-vy]
        self.pos2 = self.pos1 + self.vel

        ipx1 = int(self.pos1[0])
        ipy1 = int(self.pos1[1])
        ipx2 = int(self.pos2[0])
        ipy2 = int(self.pos2[1])
        self.im2_roi_orig = self.image2[ipx1:ipx1+self.w, \
                                    ipy1:ipy1+self.h]
        if ipx2>=0 and ipy2>=0 and ipx2+self.w<self.imw and ipy2+self.h<self.imh:
            self.im2_roi = self.image2[ipx2:ipx2+self.w, \
                                    ipy2:ipy2+self.h]
        else:
            self.im2_roi = np.zeros((self.w,self.h,self.imd))
        self.max_ll = mx
        self.fluo = self.fluorescence()
        return self.vel

    def d_fluorescence(self):
        return np.mean(self.im2_roi - self.im1_roi, axis=(0,1))

    def fluorescence(self):
        return np.mean(self.im1_roi, axis=(0,1))

    def save_rois(self, file_pattern1, file_pattern2):
        fname1 = file_pattern%(self.t1)
        fname2 = file_pattern%(self.t2)
        imsave(fname1, self.im1_roi, plugin='tifffile')
        imsave(fname1, self.im2_roi, plugin='tifffile')



class Ensemble():
    # Set of ensemble states at each time t
    def __init__(self):
        # Dictionary mapping time t to ensemble state
        self.states = {}

    def __getitem__(self, t):
        return self.states[t]

    def __setitem__(self, t, state):
        self.states[t] = state

    def pos(self):
        # Return array of ensemble positions at each time
        return np.array([s.pos1 for t,s in self.states.items()])

    def vel(self):
        # Return array of ensemble positions at each time
        return np.array([s.vel for t,s in self.states.items()])

    def max_ll(self):
        # Return array of ensemble positions at each time
        return np.array([s.max_ll for t,s in self.states.items()])

    def fluo(self):
        # Return array of ensemble positions at each time
        return np.array([s.fluorescence() for t,s in self.states.items()])

class EnsembleGrid:
    def __init__(self, images, masks):
        # Dictionary of ensembles mapped by grid position
        self.ensembles = []
        # Array of images for computing ensembles
        self.images = images
        self.masks = masks
        s = images[0].shape
        self.imw = s[0] 
        self.imh = s[1]
        if len(s)==3:
            self.imd = s[2]
        else:
            self.imd = 1
        self.nt = 0
        self.n = 0

    def __getitem__(self, ix,iy):
        return self.ensembles[(ix,iy)]

    def __getitem__(self, ix,iy,it):
        return self.ensembles[(ix,iy)][it]

    def __setitem__(self, ix,iy,ensemble):
        self.ensembles[(ix,iy)] = ensemble

    def initialise_ensembles(self, w,h, sw,sh, px0,py0, vx_max, vy_max, dt=1):
        # Initialise each ensemble with states at each time
        self.gx = int(self.imw/sw)
        self.gy = int(self.imh/sh)
        self.nt = 1
        self.gw,self.gh = w,h
        self.n = 0
        for ix in range(self.gx):
            for iy in range(self.gy):
                pos = np.array([px0+ix*sw, py0+iy*sh])
                state = EnsembleState(pos, 0, dt, \
                                        self.images[0,:,:], \
                                        self.images[1,:,:], \
                                        self.masks[0,:,:], \
                                        w,h)

                # Only add ensemble if it meets the requirements for tracking, ie. valid entropy map
                mask = True #,ll = state.entropy_map(vx_max, vy_max, nbins=nbins, hmin=hmin)
                if mask:
                    e = Ensemble()
                    e[0] = state
                    self.ensembles.append(e)
                    self.n += 1

    def compute_motion(self, nt, vx_max, vy_max, dt=1):
        self.nt = nt
        for ensemble in self.ensembles:
            state0 = ensemble.states[0]
            mask,ll = state0.entropy_map(vx_max, vy_max, nbins=nbins, hmin=hmin)
            state0.llmap = ll
            if mask:
                vel,mx = state0.compute_mean_velocity(-ll)
            else:
                #print("Skipping")
                vel = [0,0]
                mx = 0
            for t in range(1,nt):
                prevstate = ensemble.states[t-1]
                state = EnsembleState(prevstate.pos2, t, t+dt, \
                                        self.images[t,:,:], \
                                        self.images[t+1,:,:], \
                                        self.masks[t,:,:], \
                                        self.gw,self.gh)
                if prevstate.live:
                    mask,ll = state.entropy_map(vx_max=vx_max, vy_max=vy_max, nbins=nbins, hmin=hmin)
                    state.llmap = ll
                    if mask:
                        vel,mx = state.compute_mean_velocity(-ll)
                    else:
                        vel = [0,0]
                        mx = 0
                else:
                    state.live = False

                ensemble[t] = state

    def pos(self):
        pos = np.zeros((self.n,self.nt,2))
        for i in range(len(self.ensembles)):
            pos[i,:,:] = self.ensembles[i].pos()
        return pos
        
    def vel(self):
        vel = np.zeros((self.n,self.nt,2))
        for i in range(len(self.ensembles)):
            vel[i,:,:] = self.ensembles[i].vel()
        return vel

    def velmag(self):
        vel = self.vel()
        velmag = np.sqrt(vel[:,:,0]**2 + vel[:,:,1]**2)
        return velmag

    def angle(self):
        vel = self.vel()
        ang = np.arctan2(vel[:,:,0], vel[:,:,1])
        ang[ang<0] += 2*np.pi
        return ang

    def cosine(self):
        vel = self.vel()
        cos = vel[:,:,0]/np.sqrt(vel[:,:,1]**2 + vel[:,:,0]**2)
        return cos

    def max_ll(self):
        max_ll = np.zeros((self.n,self.nt))
        for i in range(len(self.ensembles)):
            max_ll[i,:] = self.ensembles[i].max_ll()
        return max_ll

    def fluo(self):
        fluo = np.zeros((self.n,self.nt,self.imd))
        for i in range(len(self.ensembles)):
            fluo[i,:,:] = self.ensembles[i].fluo().reshape((self.nt,self.imd))
        return fluo

    def save_data(self, outdir):
        fname = os.path.join(outdir, 'pos.np')
        self.pos().tofile(fname)

        fname = os.path.join(outdir, 'vel.np')
        self.vel().tofile(fname)

        fname = os.path.join(outdir, 'max_ll.np')
        self.max_ll().tofile(fname)

        fname = os.path.join(outdir, 'fluo.np')
        self.fluo().tofile(fname)

    def save_rois(self, outdir, file_pattern1, file_pattern2):
        for e in self.ensembles:
            for t,s in e.states.items():
                fname1 = os.path.join(outdir, file_pattern1%(s.pos1[0],s.pos1[1],t))
                fname2 = os.path.join(outdir, file_pattern2%(s.pos1[0],s.pos1[1],t))
                # Assume the image is 16 bit tiff
                if np.max(s.im1_roi)>0:
                    imsave(fname1, s.im1_roi.astype(np.uint16), plugin='tifffile')
                if np.max(s.im2_roi_orig)>0:
                    imsave(fname2, s.im2_roi.astype(np.uint16), plugin='tifffile')

    def save_llmap(self, outdir, file_pattern):
        for e in self.ensembles:
            for t,s in e.states.items():
                fname = os.path.join(outdir, file_pattern%(s.pos1[0],s.pos1[1],t))
                # Assume the image is 16 bit tiff
                if np.max(s.weight)>0:
                    imsave(fname, s.weight.astype(np.float32), plugin='tifffile')
                #s.llmap.tofile(fname)
                #print(np.max(s.llmap))

    def save_quivers(self, outdir, file_pattern1, file_pattern2, normed=False):
        plt.figure(figsize=(12,12))
        pos = self.pos()
        vel = self.vel()
        for t in range(self.nt):
            im = self.images[t]
            mask = self.masks[t]
            #plt.imshow(im/np.max(im), origin='lower' )
            plt.imshow(mask, origin='lower' )
            if normed:
                norm = np.sqrt(vel[:,t,0]**2 + vel[:,t,1]**2)
            else:
                norm = 1
            plt.quiver(self.gw/2+pos[:,t,1], \
                        self.gh/2+pos[:,t,0], \
                        vel[:,t,1]/norm, \
                        vel[:,t,0]/norm, \
                        scale=5, \
                        scale_units='inches')
            fname = os.path.join(outdir, file_pattern1%(t))
            plt.savefig(fname)
            plt.clf()
            plt.quiver(self.gw/2+pos[:,t,1],self.gh/2+pos[:,t,0], vel[:,t,1]/norm, vel[:,t,0]/norm)
            fname = os.path.join(outdir, file_pattern2%(t))
            plt.savefig(fname)
            plt.clf()

    def save_paths(self, outdir, file_pattern1, file_pattern2):
        plt.figure(figsize=(12,12))
        pos = self.pos()
        for t in range(self.nt):
            im = self.images[t]
            mask = self.masks[t]
            plt.imshow(im/np.max(im), origin='lower' )
            #plt.imshow(mask, origin='lower' )
            for tt in range(t):
                plt.plot(self.gw/2+pos[:,tt,1], self.gh/2+pos[:,tt,0],'w.')
            plt.plot(self.gw/2+pos[:,t,1], self.gh/2+pos[:,t,0],'r.')
            fname = os.path.join(outdir, file_pattern1%(t))
            plt.savefig(fname)
            plt.clf()
            for tt in range(t):
                plt.plot(self.gw/2+pos[:,tt,1], self.gh/2+pos[:,tt,0],'k.')
            plt.plot(self.gw/2+pos[:,t,1], self.gh/2+pos[:,t,0],'r.')
            fname = os.path.join(outdir, file_pattern2%(t))
            plt.savefig(fname)
            plt.clf()

