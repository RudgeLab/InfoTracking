import numpy as np
import skimage
from skimage.io import imsave
import infotracking.infotheory as infotheory
import matplotlib
import matplotlib.pyplot as plt
import os
from skimage.filters import gaussian,median
from numpy.fft import fft2
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fmin, minimize

class EnsembleState():
    def __init__(self, imw, imh, pos1, t1, t2, prev_vel, w=64, h=64):
        # Ensemble computed from image pair, stores regions of interest at t and t+1
        self.t1 = t1
        self.t2 = t2
        self.w = w
        self.h = h
        self.pos1 = pos1
        self.pos2 = pos1
        self.prev_vel = prev_vel
        self.vel = [0,0]
        self.in_mask = False
        #self.im1_roi = None
        #self.im2_roi = None
        self.max_ll = 0
        # Size of images
        self.imw = imw
        self.imh = imh
        self.imd = 1
        self.im1_roi = np.zeros((self.w,self.h,self.imd))
        self.im2_roi = np.zeros((self.w,self.h,self.imd))


    def entropy_map(self, image1, 
            image2, 
            mask, 
            init_vel,
            vx_max=7, vy_max=7, 
            nbins=16, 
            hmax=4.0,
            mask_threshold=0.5):
        # Compute conditional entropy H(I2|I1) at each displacement
        vw = vx_max*2 + 1
        vh = vy_max*2 + 1
        log_likelihood = np.zeros((vw,vh))

        # ROI in first image
        ipx1 = int(self.pos1[0])
        ipy1 = int(self.pos1[1])
        # If ROI of first image falls outside image boundary return empty map
        # and set empty ROIs
        if ipx1<0 or ipy1<0 or ipx1+self.w>=self.imw or ipy1+self.h>=self.imh:
            #print('Outside image')
            return False, log_likelihood
        # Check mask
        self.mask_roi = mask[ipx1:ipx1+self.w, \
                                    ipy1:ipy1+self.h]
        mask_content = np.mean(self.mask_roi)
        if mask_content<mask_threshold:
        #if self.mask_roi[int(self.w/2),int(self.h/2)]==0:
            # Outside colony
            print('Outside colony')
            return False, log_likelihood
        else:
            self.in_mask = True
            #return True, log_likelihood

        # Analyse image region
        self.im1_roi = image1[ipx1:ipx1+self.w, \
                                    ipy1:ipy1+self.h]

        # Entropy of image region, if too low cannot compute velocity
        hgram, edges = np.histogram( self.im1_roi.ravel(), \
            bins=nbins)
        hx = infotheory.entropy(hgram)
        # Store entropy for debugging
        self.hx = hx
        if hx>hmax:
            # Image region has low entropy
            print('Image entropy too high')
            return False, log_likelihood

        # Initial guess of velocity to centre entropy map
        self.vinit = init_vel[ipx1:ipx1+self.w, ipy1:ipy1+self.h, :].mean(axis=(0,1))


        for vx in range(-vx_max,vx_max+1):
            for vy in range(-vx_max,vy_max+1):
                ipx2 = int(self.pos1[0] + vx + self.vinit[0])
                ipy2 = int(self.pos1[1] + vy + self.vinit[1])
                # Check roi is inside image, otherwise entropy is zero
                if ipx2>=0 and ipy2>=0 and ipx2+self.w<self.imw and ipy2+self.h<self.imh:
                    im2_roi = image2[ipx2:ipx2+self.w, \
                                                ipy2:ipy2+self.h]
                    hgram_offset, xedges, yedges = np.histogram2d( self.im1_roi.ravel(), im2_roi.ravel(), \
                                                        bins=nbins)
                    #hy_cond_x_val  = infotheory.conditional_entropy(hgram_offset, ax=1)
                    mi  = infotheory.mutual_information(hgram_offset)
                    log_likelihood[vx+vx_max,vy+vy_max] = mi
        # Return the map of conditional entropy at each offset
        return True,log_likelihood

    def compute_mean_velocity(self, 
            image1, 
            image2, 
            mask, 
            init_vel,
            llmap, 
            velstd, 
            threshold=0.75):
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

        pky,pkx = np.meshgrid(np.arange(-ww,ww+1), np.arange(-hh,hh+1))
        #self.weight = np.exp(llmap)  * np.exp(-(pkx**2 + pky**2)/(2*velstd**2))
        self.weight = llmap  -(pkx**2 + pky**2)/(2*velstd**2)
        #plt.imshow(self.weight)
        #plt.colorbar()
        #plt.show()

        print('Setting llmap')
        self.llmap = llmap

        init_peakx,init_peaky = np.where(self.weight==np.max(self.weight))
        if len(init_peakx)==0 or len(init_peaky)==0:
            return [0,0]
        print(init_peakx, init_peaky)
        init_peakx = init_peakx[0]
        init_peaky = init_peaky[0]
        initx = pkx[init_peakx,init_peaky]
        inity = pky[init_peakx,init_peaky]
        
        rspl = RectBivariateSpline(pkx[:,0], pky[0,:], self.weight, kx=2,ky=2)
        def rsplfunc(x):
            return -rspl(x[0],x[1])
        minx = fmin(rsplfunc, [initx,inity], disp=False)
        rspl = RectBivariateSpline(pkx[:,0], pky[0,:], llmap, kx=2,ky=2)
        vx,vy = minx
        print('fmin found solution:')
        print(minx)
        print('Log likelihood at minimum:')
        #print(sol.fun)
        print('from initial guess:')
        print(initx,inity)
        
        #vx,vy = initx,inity
        #plt.imshow(self.im1_roi)
        #plt.imshow(self.weight)
        #plt.plot(init_peaky,init_peakx,'rx')
        #plt.show()

        # Store velocity, image roi in next time step, and maximum log likelihood
        if abs(vx)>ww or abs(vy)>hh:
            print('Velocity out of bounbds: ', vx, vy)
            vx,vy = 0,0
            self.max_ll = rspl(0, 0) # np.max(self.llmap)
        else:
            self.max_ll = rspl(minx[0], minx[1]) # np.max(self.llmap)
        print('max_ll ', self.max_ll)
        self.vel = [vx + self.vinit[0],vy + self.vinit[1]]
        print(f'Computed velocity = {self.vel}')
        self.pos2 = self.pos1 + self.vel

        ipx1 = int(self.pos1[0])
        ipy1 = int(self.pos1[1])
        ipx2 = int(self.pos2[0])
        ipy2 = int(self.pos2[1])
        self.im2_roi_orig = image2[ipx1:ipx1+self.w, \
                                    ipy1:ipy1+self.h]
        if ipx2>=0 and ipy2>=0 and ipx2+self.w<self.imw and ipy2+self.h<self.imh:
            self.im2_roi = image2[ipx2:ipx2+self.w, \
                                    ipy2:ipy2+self.h]
        else:
            self.im2_roi = np.zeros((self.w,self.h,self.imd))


        '''
        # Plots for debugging
        plt.subplot(1,4,1)
        print(self.vel)
        plt.imshow(self.weight)
        plt.plot(vy + hh, vx + ww,  'r+')
        plt.colorbar()
        plt.subplot(1,4,2)
        plt.imshow(self.im1_roi)
        plt.subplot(1,4,3)
        plt.imshow(self.im2_roi_orig)
        plt.subplot(1,4,4)
        plt.imshow(self.im2_roi)
        plt.show()
        '''
        return self.vel

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

    def pos2(self):
        # Return array of ensemble positions at each time
        return np.array([s.pos2 for t,s in self.states.items()])

    def vel(self):
        # Return array of ensemble positions at each time
        return np.array([s.vel for t,s in self.states.items()])

    def hx(self):
        # Return array of ensemble entropy at each time
        return np.array([s.hx for t,s in self.states.items()])

    def max_ll(self):
        # Return array of ensemble positions at each time
        return np.array([s.max_ll for t,s in self.states.items()])

    def mask(self):
        # Return array of ensemble mask flags at each time (True if ensemble is inside colony)
        return np.array([s.in_mask for t,s in self.states.items()])


class EnsembleGrid:
    def __init__(self, images, masks, init_vel, mask_threshold):
        # Dictionary of ensembles mapped by grid position
        self.ensembles = {} 
        # Array of images for computing ensembles
        self.images = images
        self.masks = masks
        self.init_vel = init_vel
        self.mask_threshold = mask_threshold
        s = images[0].shape
        self.imw = s[0] 
        self.imh = s[1]
        if len(s)==3:
            self.imd = s[2]
        else:
            self.imd = 1
        self.nt = 0
        self.gx, self.gy = 0,0

    def __getitem__(self, ix,iy):
        return self.ensembles[(ix,iy)]

    def __getitem__(self, ix,iy,it):
        return self.ensembles[(ix,iy)][it]

    def __setitem__(self, ix,iy,ensemble):
        self.ensembles[(ix,iy)] = ensemble

    def initialise_ensembles(self, w,h, sw,sh, px0,py0, dt=1):
        # Initialise each ensemble with states at each time
        self.gx = int(self.imw/sw)
        self.gy = int(self.imh/sh)
        self.nt = 1
        self.gw,self.gh = w,h
        self.sw,self.sh = sw,sh
        for ix in range(0, self.gx):
            for iy in range(0, self.gy):
                pos = np.array([px0+ix*sw, py0+iy*sh])
                self.ensembles[(ix,iy)] = Ensemble()
                state = EnsembleState(self.imw,self.imh, 
                                        pos1=pos, t1=0, t2=dt, \
                                        prev_vel=np.array([0,0]),
                                        w=w, h=h)
                self.ensembles[(ix,iy)][0] = state

    def compute_motion(self, nt, vx_max, vy_max, velstd, dt=1):
        self.nt = nt
        n = self.gx*self.gy
        i = 0
        for (ix,iy),ensemble in self.ensembles.items():
            print('Computing grid square (%d,%d),  %d of %d'%(ix,iy,i,n))
            i += 1
            state0 = ensemble.states[0]
            mask,ll = state0.entropy_map(self.images[0,:,:], \
                                       self.images[1,:,:], 
                                       self.masks[0,:,:], \
                                        self.init_vel[0,:,:,:],
                                       vx_max, vy_max, nbins=16, hmax=1e10,
                                       mask_threshold=self.mask_threshold)
            if mask:
                vel,mx = state0.compute_mean_velocity(self.images[0,:,:], \
                                        self.images[1,:,:], 
                                        self.masks[0,:,:], 
                                        self.init_vel[0,:,:,:],
                                        ll, velstd)
            else:
                print('Outside mask ', (ix,iy))
                state0.vel = [np.nan,np.nan]
                mx = 0

            for t in range(1,nt):
                prevstate = ensemble.states[t-1]
                state = EnsembleState(self.imw,self.imh, pos1=prevstate.pos1, t1=t, t2=t+dt, \
                                        prev_vel=prevstate.vel,
                                        w=self.gw, h=self.gh)
                mask = False 
                mask,ll = state.entropy_map(self.images[t,:,:], \
                                        self.images[t+1,:,:], 
                                        self.masks[t,:,:], \
                                        self.init_vel[t,:,:,:],
                                        vx_max, vy_max, nbins=16, hmax=1e10,
                                        mask_threshold=self.mask_threshold)
                if mask:
                    vel,mx = state.compute_mean_velocity(self.images[t,:,:], \
                                        self.images[t+1,:,:], 
                                        self.masks[t,:,:], 
                                        self.init_vel[t,:,:,:],
                                        ll, velstd)
                else:
                    print('Outside mask ', (ix,iy))
                    state.vel = [np.nan,np.nan]
                    mx = 0
                ensemble[t] = state

    def mask(self):
        mask = np.zeros((self.gx,self.gy,self.nt))
        for (ix,iy),e in self.ensembles.items():
            mask[ix,iy,:] = e.mask()
        return mask
        
    def pos(self):
        pos = np.zeros((self.gx,self.gy,self.nt,2))
        for (ix,iy),e in self.ensembles.items():
            pos[ix,iy,:,:] = e.pos()
        return pos
        
    def pos2(self):
        pos2 = np.zeros((self.gx,self.gy,self.nt,2))
        for (ix,iy),e in self.ensembles.items():
            pos2[ix,iy,:,:] = e.pos2()
        return pos2
        
    def vel(self):
        vel = np.zeros((self.gx,self.gy,self.nt,2))
        for (ix,iy),e in self.ensembles.items():
            vel[ix,iy,:,:] = e.vel()
        #for i in range(self.nt):
        #    vel[:,:,i,0] = gaussian(vel[:,:,i,0], 1.)
        #    vel[:,:,i,1] = gaussian(vel[:,:,i,1], 1.)
        return vel

    def velmag(self):
        vel = self.vel()
        velmag = np.sqrt(vel[:,:,:,0]**2 + vel[:,:,:,1]**2)
        return velmag

    def angle(self):
        vel = self.vel()
        ang = np.arctan2(vel[:,:,:,0], vel[:,:,:,1])
        ang[ang<0] += 2*np.pi
        return ang

    def hx(self):
        hx = np.zeros((self.gx,self.gy,self.nt))
        for (ix,iy),e in self.ensembles.items():
            hx[ix,iy,:] = e.hx()
        return hx

    def max_ll(self):
        max_ll = np.zeros((self.gx,self.gy,self.nt))
        for (ix,iy),e in self.ensembles.items():
            max_ll[ix,iy,:] = e.max_ll()
        return max_ll

    def save_data(self, outdir):
        fname = os.path.join(outdir, 'pos.np')
        np.save(fname, self.pos())

        fname = os.path.join(outdir, 'pos2.np')
        np.save(fname, self.pos2())

        fname = os.path.join(outdir, 'vel.np')
        np.save(fname, self.vel())

        #fname = os.path.join(outdir, 'max_ll.np')
        #np.save(fname, self.max_ll())

    def save_rois(self, outdir, file_pattern):
        for (ix,iy),e in self.ensembles.items():
            for t,s in e.states.items():
                fname = os.path.join(outdir, file_pattern%(ix,iy,t))
                # Assume the image is 16 bit tiff
                imsave(fname, s.im1_roi.astype(np.uint16), plugin='tifffile')

    def save_quivers(self, outdir, file_pattern1, file_pattern2, normed=False):
        plt.figure(figsize=(24,24))
        pos = self.pos()
        vel = self.vel()
        for t in range(self.nt):
            im = self.images[t]
            plt.imshow(im/np.max(im), origin='lower' )
            if normed:
                norm = np.sqrt(vel[:,:,t,0]**2 + vel[:,:,t,1]**2)
            else:
                norm = 1
            vx = vel[:,:,t,0]
            vy = vel[:,:,t,1]
            plt.quiver(self.gw/2+pos[:,:,t,1],self.gh/2+pos[:,:,t,0], -vy/norm, -vx/norm)
            fname = os.path.join(outdir, file_pattern1%(t))
            plt.savefig(fname)
            plt.clf()
            plt.quiver(self.gw/2+pos[:,:,t,1],self.gh/2+pos[:,:,t,0], -vy/norm, -vx/norm)
            fname = os.path.join(outdir, file_pattern2%(t))
            plt.savefig(fname)
            plt.clf()

    def save_paths(self, outdir, file_pattern1, file_pattern2):
        plt.figure(figsize=(24,24))
        pos = self.pos()
        for t in range(self.nt):
            im = self.images[t]
            plt.imshow(im/np.max(im), origin='lower' )
            for tt in range(t):
                plt.plot(self.gw/2+pos[:,:,tt,1], self.gh/2+pos[:,:,tt,0],'w.')
            plt.plot(self.gw/2+pos[:,:,t,1], self.gh/2+pos[:,:,t,0],'r.')
            fname = os.path.join(outdir, file_pattern1%(t))
            plt.savefig(fname)
            plt.clf()
            for tt in range(t):
                plt.plot(self.gw/2+pos[:,:,tt,1], self.gh/2+pos[:,:,tt,0],'k.')
            plt.plot(self.gw/2+pos[:,:,t,1], self.gh/2+pos[:,:,t,0],'r.')
            fname = os.path.join(outdir, file_pattern2%(t))
            plt.savefig(fname)
            plt.clf()

