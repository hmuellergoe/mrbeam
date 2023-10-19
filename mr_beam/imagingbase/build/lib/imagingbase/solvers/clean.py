from regpy.solvers import Solver

import logging
import numpy as np
from scipy.ndimage import shift

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

        #self._startmod()

    def _startmod(self):
        self.index = self.shape//2
        self.x.append([self.index, self.strength])
        self._minor_loop()
        self.y = self.dmap.flatten()

    def _minor_loop(self):
        self.dmap -= self.gain * self.strength * shift(self.dbeam, self.index-self.shape//2)

    def _next(self):
        if self.add_only_positive_coeff:
            self.index = np.asarray( np.unravel_index( 
                self.indices[ np.argmax(self.dmap.flatten()[self.window]) ], self.shape) )
        else:
            self.index = np.asarray( np.unravel_index( 
                self.indices[ np.argmax( np.abs(self.dmap.flatten()) [self.window]) ], self.shape) )            
        self.strength = self.dmap[self.index[0], self.index[1]]
        self.strength_total += self.strength*self.gain*self.flux_dbeam
        self.x.append([self.index, self.strength, self.gain])
        self._minor_loop()
        self.counter += 1
        self.y = self.dmap.flatten()
        if (self.counter//50)*50 == self.counter:
            print(self.strength_total, 'cleaned at iteration ', self.counter)
            print('rms', np.std(self.y[self.window]))

    def finalize(self):
        cmap = self.dmap.copy()
        for item in self.x:
            cmap += self.gain*item[1]*shift(self.psf, item[0]-self.shape//2)
        return cmap
    
    def intermediate(self):
        cmap = np.zeros(self.shape)
        for item in self.x:
            cmap += self.gain*item[1]*shift(self.psf, item[0]-self.shape//2)
        self.x = []
        return cmap
    
    def update_map_list(self, dmap):
        self.dmap = dmap #/ self.flux_dbeam







