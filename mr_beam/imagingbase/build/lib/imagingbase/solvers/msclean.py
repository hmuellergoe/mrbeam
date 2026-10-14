from regpy.solvers import Solver

import logging
import numpy as np
from scipy.ndimage import shift

from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

class CLEAN(Solver):
    def __init__(self, dmap, dbeam, psf, window, gain=0.1, start=1, add_only_positive_coeff=False):
        super().__init__()
        self.gain = gain
        self.window = window.flatten()
        self.strength = start
        
        self.add_only_positive_coeff = add_only_positive_coeff
        
        self.dmap = dmap
        self.dbeam = dbeam
        self.psf = psf
        
        self.max_beam = np.max(self.dbeam)
        self.dbeam /= self.max_beam
        self.psf /= self.max_beam
        
        self.flux_dbeam = np.sum(self.dbeam)
        self.strength_total = self.strength * self.gain * self.flux_dbeam

        #if self.window == None:
        #    self.window = np.zeros(self.dmap.shape, dtype=int)

        self.shape = np.asarray( self.dbeam.shape )
        self.indices = np.arange(0, np.prod(self.shape))[self.window]

        assert self.dmap.shape == self.dbeam.shape
        assert self.psf.shape == self.dbeam.shape
        assert len(self.dbeam.shape) == 2
        #assert that image dimension is even
        #assert ( (self.shape//2)*2 == np.asarray(self.shape) ).any()

        self.x = []
        self.y = self.dmap.flatten()
        
        self.counter = 0
        
        self.basis_function = self._approximate_basis_function()
        self.msdbeam = np.convolve(self.basis_function.flatten(), self.dbeam.flatten(), mode='same').reshape(self.shape)
        max_beam = np.max(self.msdbeam)
        self.msdbeam /= max_beam
        self.basis_function /= max_beam
        #self.search_msdbeam = np.convolve(self.basis_function.flatten(), self.msdbeam.flatten(), mode='same').reshape(self.shape)
        self.search_map = np.convolve(self.basis_function.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)


        #self._startmod()
        
    def _minor_loop(self):
        #self.search_map -= self.gain * self.strength * shift(self.msdbeam, self.index-self.shape//2)
        self.dmap -= self.gain * self.strength * shift(self.msdbeam, self.index-self.shape//2)
        #self.search_map -= self.gain * self.strength * shift(self.search_msdbeam, self.index-self.shape//2)

    def _approximate_basis_function(self):
        corr = np.correlate(self.dmap.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)
        self.basis_dbeam = np.convolve(self.dbeam.flatten(), self.dbeam.flatten(), mode='same').reshape(self.shape)
        self.basis_cbeam = np.convolve(self.psf.flatten(), self.psf.flatten(), mode='same').reshape(self.shape)
        
        dmap = corr.copy()
        
        #dmap = self.dmap.copy()
        #dbeam = self.dbeam.copy()
        #cbeam = self.psf.copy()
        
        #indices = np.arange(0, np.prod(self.shape))
        
        max_beam = np.max(self.basis_dbeam)
        self.basis_dbeam /= max_beam
        self.basis_cbeam /= max_beam
        
        gain = 0.1
        
        self.basis_x = []
        #Get a good representation of the correlation matrix
        for i in range(10):
            index = np.asarray( np.unravel_index( self.indices[ np.argmax(dmap.flatten()[self.window]) ], self.shape) )
            strength = dmap[index[0], index[1]]
            dmap -= gain * strength * shift(self.basis_dbeam, index-self.shape//2)
            self.basis_x.append([index, strength, gain])
        
        self.cmap = np.zeros(self.shape)
        self.mod = np.zeros(self.shape)
        self.delta = np.zeros(self.shape)
        self.delta[self.shape[0]//2, self.shape[1]//2] = 1.
        for item in self.basis_x:
            self.cmap += gain*item[1]*shift(self.delta, item[0]-self.shape//2)#shift(self.basis_cbeam, item[0]-self.shape//2)
            self.mod += gain*item[1]*shift(self.basis_dbeam, item[0]-self.shape//2)
         
        return self.cmap #+ dmap    
        #return np.real(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cmap))))
            
        #return cmap
        #x, y = np.meshgrid(np.arange(-self.shape[0]//2, self.shape[0]//2+1), np.arange(-self.shape[1]//2, self.shape[1]//2+1))
        
    def _update_basis_function(self):
        corr = np.correlate(self.dmap.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)
        dmap = corr - self.mod
        
        gain = 0.1
        #indices = np.arange(0, np.prod(self.shape))
        
        for i in range(10):
            index = np.asarray( np.unravel_index( self.indices[ np.argmax(np.abs(dmap).flatten()[self.window]) ], self.shape) )
            strength = dmap[index[0], index[1]]
            dmap -= gain * strength * shift(self.basis_dbeam, index-self.shape//2)
            self.basis_x.append([index, strength, gain])
            
            self.cmap += gain*strength*shift(self.delta, index-self.shape//2)#shift(self.basis_cbeam, index-self.shape//2)
            self.mod += gain*strength*shift(self.basis_dbeam, index-self.shape//2)
        
        return self.cmap #+ dmap
        
        
#or: this is the covariance matrix for a Gaussian, wouldn't be anything different than fitting a Gaussian to the residual, that has been done already 
#note that gaussian with this std would not have the correct correlation, correlation of two gaussians is gaussian, so we would fit a gaussian to cmap first to get the correct gaussian with correct cross-correlation 
#also sigma is 2x2 matrix, do not confuse with Wiener-Khinchin theorem which would give a correctly correlated distribution (that is a completely random field and not just a beam/basis function)
#still an idea to consider: model the image with Gaussian field, Wiener Khinchin: Fourier, multiply with corr, Fourier back, now we do CLEAN like delta functions, so multiplication with correlation translates into a convolution --> basis function
        
        #u, s, v = np.linalg.svd(corr, hermitian=True)
        #mask = np.zeros(s.shape)
        #mask[0] = 1
        #w = u @ np.diag(s*mask) @ v
##############        
        #downsample = 16
        #corr = np.outer(self.dmap[::downsample,::downsample].flatten(), self.dmap[::downsample,::downsample].flatten())
        #v, w = largest_eigh(corr, subset_by_index=[corr.shape[0]-1, corr.shape[0]-1])
##############

#        patch_size = 64

#        patches = []
#        for i in range(self.shape[0]//patch_size*2-1):
#            for j in range(self.shape[0]//patch_size*2-1):
#                patches.append(self.dmap[i*patch_size//2:(i+2)*patch_size//2+1,j*patch_size//2:(j+2)*patch_size//2+1])
#                
#        patches = np.array(patches)
#        
#        #to_sum = np.zeros((patches[0].shape[0]*patches[0].shape[1], patches[0].shape[0]*patches[0].shape[1]))
#
#
#        #outers = []
#        corrs = []
#        for i in range(patches.shape[0]):
#        #    #to_sum += np.outer(patches[i].flatten(), patches[i].flatten()).reshape((patches[0].shape[0]*patches[0].shape[1], patches[0].shape[0]*patches[0].shape[1]))
#        #    outers.append(np.outer(patches[i].flatten(), patches[i].flatten()).reshape((patches[0].shape[0]*patches[0].shape[1], patches[0].shape[0]*patches[0].shape[1])))
#            corrs.append(np.correlate(patches[i].flatten(), patches[i].flatten(), mode='same'))
#
#
#        #outers = np.array(outers)
#        #corrs = np.array(corrs)
#        
#        #corr = patches.reshape((patches.shape[0], patches.shape[1]*patches.shape[2]))
#        #XTX = corr.transpose() @ corr
#        
#        XTX = corrs.transpose() @ corrs
#        
#        v, w = largest_eigh(XTX, subset_by_index=[XTX.shape[0]-1, XTX.shape[0]-1])
#       
#        #mean = to_sum / patches.shape[0]
#        
#        #v, w = largest_eigh(mean, subset_by_index=[mean.shape[0]-1, mean.shape[0]-1])
        
    def _next(self):
        #find maximum
        #self.search_map = np.convolve(self.basis_function.flatten(), self.dmap, mode='same').reshape(self.shape)
        if self.add_only_positive_coeff:
            self.index = np.asarray( np.unravel_index( 
                self.indices[ np.argmax(self.search_map.flatten()[self.window]) ], self.shape) )
        else:
            self.index = np.asarray( np.unravel_index( 
                self.indices[ np.argmax( np.abs(self.search_map.flatten()) [self.window]) ], self.shape) )            
        
        #append list
        self.strength = self.dmap[self.index[0], self.index[1]]
        self.strength_total += self.strength*self.gain*self.flux_dbeam
        self.x.append([self.index, self.strength, self.gain, self.basis_function])
        
        #update residual
        self._minor_loop()
        self.counter += 1
        self.y = self.dmap.flatten()
        
        self.basis_function = self._update_basis_function()
        self.msdbeam = np.convolve(self.basis_function.flatten(), self.dbeam.flatten(), mode='same').reshape(self.shape)
        max_beam = np.max(self.msdbeam)
        self.msdbeam /= max_beam
        self.basis_function /= max_beam
        #self.search_msdbeam = np.convolve(self.basis_function.flatten(), self.msdbeam.flatten(), mode='same').reshape(self.shape)
        self.search_map = np.convolve(self.basis_function.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)
    

#search for maximum: we need to convolve to find the biggest correlation
#strength: the value at that position unconvolved
#substitute: dbeam convolved with basis function
#for this the basis function needs to be normalized such that dbeam*basis_function has maximum 1
#do this in approximate_basis_function


#maybe we need to do the evaulation of the basis function more often at the beginning
#since the basis functions get much more regular at later iterations

    def startmod(self):
        #Make some iterations first with iteratively approximating the basis_function
        dmap_old = self.dmap.copy()
        for i in range(10):
            self._next()
            self.counter = 0
            
        #Now set all parameters to their initial values again    
        self.dmap = dmap_old
        self.x = []
        self.y = self.dmap.flatten()
        
        #We set the counter to one, to avoid initial recalculation
        self.counter = 1
        
        #Reinitialize the multiscale products
        self.msdbeam = np.convolve(self.basis_function.flatten(), self.dbeam.flatten(), mode='same').reshape(self.shape)
        max_beam = np.max(self.msdbeam)
        self.msdbeam /= max_beam
        self.basis_function /= max_beam
        self.search_msdbeam = np.convolve(self.basis_function.flatten(), self.msdbeam.flatten(), mode='same').reshape(self.shape)
        self.search_map = np.convolve(self.basis_function.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)
        
    def finalize(self):
        cmap = np.zeros(self.shape)
        for item in self.x:
            cmap += item[2]*item[1]*shift(item[3], item[0]-self.shape//2)
        return cmap


#clean delta components first, then the extended structure
#or probably do this in parallel, i.e. decide during the algorithm
#


















