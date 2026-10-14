from regpy.solvers import Solver

import numpy as np
from scipy.ndimage import shift

class CLEAN(Solver):
    def __init__(self, dmap, dbeam, psf, window=None, window_corr=None, gain=0.1, start=1, add_only_positive_coeff=False, **kwargs):
        super().__init__()
        self.gain = gain
        self.window = window.flatten()
        self.window_corr = window_corr.flatten()
        self.strength = start
        
        self.add_only_positive_coeff = add_only_positive_coeff
        
        self.dmap = dmap
        self.dbeam = dbeam
        self.psf = psf
        
        self.max_beam = np.max(self.dbeam)
        self.dbeam /= self.max_beam
        self.psf /= self.max_beam
        
        self.flux_dbeam = np.sum(self.dbeam)

        self.shape = np.asarray( self.dbeam.shape )
        self.indices = np.arange(0, np.prod(self.shape))[self.window]
        self.indices_corr = np.arange(0, np.prod(self.shape))[self.window_corr]

        assert self.dmap.shape == self.dbeam.shape
        assert self.psf.shape == self.dbeam.shape
        assert len(self.dbeam.shape) == 2

        self.x = []
        self.y = self.dmap.flatten()
        
        self.counter = 0
        
        self.cmap = np.zeros(self.shape)
        self.mod = np.zeros(self.shape)
        self.delta = np.zeros(self.shape)
        self.delta[self.shape[0]//2, self.shape[1]//2] = 1.
        
        self.clean_map = np.zeros(self.shape)
        
        self.kwargs = kwargs
        
        if 'basis_dbeam' in self.kwargs:
            self.BB = self.kwargs.get('basis_dbeam') / self.max_beam**2
        else:
            #self.BB = np.zeros(self.shape)
            #self.BB[32:97,32:97] = np.convolve(self.dbeam[32:97,32:97].flatten(), self.dbeam[32:97,32:97].flatten(), mode='same').reshape((65,65))
            self.BB = np.convolve(self.dbeam.flatten(), self.dbeam.flatten(), mode='same').reshape(self.shape)
        max_beam = np.max(self.BB)
        self.BB /= max_beam
        self.dbeam /= np.sqrt(max_beam)
        self.psf /= np.sqrt(max_beam)
        self.max_beam *= np.sqrt(max_beam)
        #self.BB_norm = self.BB / max_beam
        
        if 'basis_dmap' in self.kwargs:
            self.tildeII = self.kwargs.get('basis_dmap')
        else:
            self.tildeII = np.correlate(self.dmap.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)
        
        if 'dirty_dmap' in self.kwargs:
            self.BI = self.kwargs.get('dirty_dmap')
        else:
            self.BI = np.convolve(self.dbeam.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)
        
        self.basis_function = self._approximate_basis_function()
        
        self.tildeMB = np.zeros(self.shape)
        self.mspsf = np.zeros(self.shape)
        #self.MB = np.zeros(self.shape)
        for item in self.basis_x:
            self.tildeMB += item[2]*item[1]*shift(self.dbeam, -(item[0]-self.shape//2))
            #self.MB += item[2]*item[1]*shift(self.dbeam, item[0]-self.shape//2)            
            self.mspsf += item[2]*item[1]*shift(self.psf, item[0]-self.shape//2)
            
        #max_msbeam = 1#np.max(self.tildeMB)
        #self.tildeMB /= max_msbeam
        #self.basis_function /= max_msbeam   
        
        self.tildeMI = np.zeros(self.shape)
        for item in self.basis_x:
            self.tildeMI += item[2]*item[1]*shift(self.dmap, -(item[0]-self.shape//2)) #/ max_msbeam
            
        self.tildeMBB = np.zeros(self.shape)
        #self.MBB = np.zeros(self.shape)
        for item in self.basis_x:
            self.tildeMBB += item[2]*item[1]*shift(self.BB, -(item[0]-self.shape//2)) #/ max_msbeam
            #self.MBB += item[2]*item[1]*shift(self.BB, item[0]-self.shape//2)
        
        self.MtildeMBB = np.zeros(self.shape)
        self.tildeMBI = np.zeros(self.shape)
        self.MtildeMB = np.zeros(self.shape)
        for item in self.basis_x:
            self.MtildeMBB += item[2]*item[1]*shift(self.tildeMBB, item[0]-self.shape//2) #/ max_msbeam
            self.tildeMBI += item[2]*item[1]*shift(self.BI, -(item[0]-self.shape//2)) #/ max_msbeam
            self.MtildeMB += item[2]*item[1]*shift(self.tildeMB, item[0]-self.shape//2) #/ max_msbeam
        
        
        '''
        self.tildeMB = np.convolve(self.basis_function.flatten(), self.dbeam.flatten(), mode='same').reshape(self.shape)
        max_msbeam = np.max(self.tildeMB)
        self.tildeMB /= max_msbeam
        self.basis_function /= max_msbeam
        self.tildeMI = np.convolve(self.basis_function.flatten(), self.dmap.flatten(), mode='same').reshape(self.shape)
        self.tildeMBB = np.convolve(self.dbeam.flatten(), self.tildeMB.flatten(), mode='same').reshape(self.shape)
        self.MtildeMBB = np.convolve(self.basis_function.flatten(), self.tildeMBB.flatten(), mode='same').reshape(self.shape) #here was a typo
        self.tildeMBI = np.convolve(self.basis_function.flatten(), self.BI.flatten(), mode='same').reshape(self.shape)
        self.MtildeMB = np.convolve(self.basis_function.flatten(), self.tildeMB.flatten(), mode='same').reshape(self.shape)
        '''
        
    def _approximate_basis_function(self):     
        gain = 0.1
        
        dmap = self.tildeII.copy()
        
        self.basis_x = []
        #Get a good representation of the correlation matrix
        for i in range(1):
            index = np.asarray( np.unravel_index( self.indices_corr[ np.argmax(dmap.flatten()[self.window_corr]) ], self.shape) )
            strength = dmap[index[0], index[1]]
            dmap -= gain * strength * shift(self.BB, index-self.shape//2)
            self.basis_x.append([index, strength, gain])
        
        for item in self.basis_x:
            self.cmap += gain*item[1]*shift(self.delta, item[0]-self.shape//2)
            self.mod += gain*item[1]*shift(self.BB, item[0]-self.shape//2)
            
        #self.tildeII += self.mod
         
        return self.cmap.copy()    

    def _next(self):
        #power_hogbom = np.linalg.norm(self.BI.flatten()[self.window])/np.linalg.norm(self.dbeam.flatten()[self.window])
        #power_ms = np.linalg.norm(self.tildeMBI.flatten()[self.window])/np.linalg.norm(self.tildeMB.flatten()[self.window])
        
        power_hogbom = np.linalg.norm(self.dmap.flatten()[self.window])
        power_ms = np.linalg.norm(self.tildeMI.flatten()[self.window])/np.linalg.norm(self.basis_function.flatten()[self.window])
        
        if power_ms > power_hogbom:
            #print('ms')
            self.temp_gain = self.gain / np.max(self.tildeMB)
            
            #find maximum
            if self.add_only_positive_coeff:
                self.index = np.asarray( np.unravel_index( 
                    self.indices[ np.argmax(self.tildeMI.flatten()[self.window]) ], self.shape) )
            else:
                self.index = np.asarray( np.unravel_index( 
                    self.indices[ np.argmax( np.abs(self.tildeMI.flatten()) [self.window]) ], self.shape) )            
            
            #append list
            if np.abs(self.dmap[self.index[0], self.index[1]]) > np.abs(self.tildeMI[self.index[0], self.index[1]]/np.max(self.MtildeMB)*np.max(self.tildeMB)):
                self.strength = self.dmap[self.index[0], self.index[1]]
            else:
                self.strength = self.tildeMI[self.index[0], self.index[1]]/np.max(self.MtildeMB)*np.max(self.tildeMB)
            #self.strength = self.dmap[self.index[0], self.index[1]]
            self.x.append([self.index, self.strength, self.temp_gain, self.basis_function])
            
            #update residual
            self._minor_loop()
            self.counter += 1
            self.y = self.dmap.flatten()
            
        else:
            #print("hogbom")
            self.temp_gain = self.gain / np.max(self.dbeam)
            
            #find maximum
            if self.add_only_positive_coeff:
                self.index = np.asarray( np.unravel_index( 
                    self.indices[ np.argmax(self.dmap.flatten()[self.window]) ], self.shape) )
            else:
                self.index = np.asarray( np.unravel_index( 
                    self.indices[ np.argmax( np.abs(self.dmap.flatten()) [self.window]) ], self.shape) )            
            
            #append list
            self.strength = self.dmap[self.index[0], self.index[1]]
            #self.strength = self.dmap[self.index[0], self.index[1]]
            self.x.append([self.index, self.strength, self.temp_gain, self.delta])
            
            #update residual
            self._minor_loop_hogbom()
            self.counter += 1
            self.y = self.dmap.flatten()
            
        self.basis_function = self._update_basis_function()
        #Now update all terms depending on the basis function, first the beams
        self.MtildeMB += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.tildeMB, self.basis_x[-1][0]-self.shape//2) + self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.tildeMB, self.basis_x[-1][0]-self.shape//2)[::-1,::-1] + (self.basis_x[-1][2]*self.basis_x[-1][1])**2*self.dbeam
        self.MtildeMBB += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.tildeMBB, self.basis_x[-1][0]-self.shape//2) + self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.tildeMBB, self.basis_x[-1][0]-self.shape//2)[::-1,::-1] + (self.basis_x[-1][2]*self.basis_x[-1][1])**2*self.BB
        self.tildeMB += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.dbeam, -(self.basis_x[-1][0]-self.shape//2))
        #self.MB += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.dbeam, self.basis_x[-1][0]-self.shape//2)
        self.mspsf += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.psf, self.basis_x[-1][0]-self.shape//2)
        self.tildeMBB += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.BB, -(self.basis_x[-1][0]-self.shape//2))
        #self.MBB += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.BB, self.basis_x[-1][0]-self.shape//2)
        
        #Now the maps
        self.tildeMI += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.dmap, -(self.basis_x[-1][0]-self.shape//2))
        self.tildeMBI += self.basis_x[-1][2]*self.basis_x[-1][1]*shift(self.BI, -(self.basis_x[-1][0]-self.shape//2))
        
        #Do the same with the conjugate
        #Now update all terms depending on the basis function, first the beams
        self.MtildeMB += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.tildeMB, self.basis_x[-2][0]-self.shape//2) + self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.tildeMB, self.basis_x[-2][0]-self.shape//2)[::-1,::-1] + (self.basis_x[-2][2]*self.basis_x[-2][1])**2*self.dbeam
        self.MtildeMBB += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.tildeMBB, self.basis_x[-2][0]-self.shape//2) + self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.tildeMBB, self.basis_x[-2][0]-self.shape//2)[::-1,::-1] + (self.basis_x[-2][2]*self.basis_x[-2][1])**2*self.BB
        self.tildeMB += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.dbeam, -(self.basis_x[-2][0]-self.shape//2))
        #self.MB += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.dbeam, self.basis_x[-2][0]-self.shape//2)
        self.mspsf += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.psf, self.basis_x[-2][0]-self.shape//2)
        self.tildeMBB += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.BB, -(self.basis_x[-2][0]-self.shape//2))
        #self.MBB += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.BB, self.basis_x[-2][0]-self.shape//2)
        
        #Now the maps
        self.tildeMI += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.dmap, -(self.basis_x[-2][0]-self.shape//2))
        self.tildeMBI += self.basis_x[-2][2]*self.basis_x[-2][1]*shift(self.BI, -(self.basis_x[-2][0]-self.shape//2))
        
        #max_msbeam = 1#np.max(self.tildeMB)
        #self.tildeMB /= max_msbeam
        #self.basis_function /= max_msbeam
        #self.tildeMBB /= max_msbeam
        #self.MtildeMBB /= (max_msbeam)**2
        #self.tildeMI /= max_msbeam
        #self.tildeMBI /= max_msbeam
        
    def _minor_loop(self):
        #self.dmap -= self.temp_gain * self.strength * shift(self.MB, self.index-self.shape//2)
        self.dmap -= self.temp_gain * self.strength * shift(self.tildeMB, self.index-self.shape//2)
        self.tildeMI -= self.temp_gain * self.strength * shift(self.MtildeMB, self.index-self.shape//2)
        
        self.tildeII += (self.temp_gain * self.strength)**2 * self.MtildeMBB - (self.temp_gain * self.strength) * shift(self.tildeMBI, -(self.index-self.shape//2))[::-1,::-1] - (self.temp_gain * self.strength) * shift(self.tildeMBI, -(self.index-self.shape//2))
        self.tildeMBI -= self.temp_gain * self.strength * shift(self.MtildeMBB, self.index-self.shape//2)
        #self.BI -= self.temp_gain * self.strength * shift(self.MBB, self.index-self.shape//2)
        self.BI -= self.temp_gain * self.strength * shift(self.tildeMBB, self.index-self.shape//2)

        self.clean_map += self.temp_gain * self.strength * shift(self.mspsf, self.index-self.shape//2)

    def _minor_loop_hogbom(self):
        self.dmap -= self.temp_gain * self.strength * shift(self.dbeam, self.index-self.shape//2)
        self.tildeMI -= self.temp_gain * self.strength * shift(self.tildeMB, self.index-self.shape//2)
        
        self.tildeII += (self.temp_gain * self.strength)**2 * self.BB - (self.temp_gain * self.strength) * shift(self.BI, -(self.index-self.shape//2))[::-1,::-1] - (self.temp_gain * self.strength) * shift(self.BI, -(self.index-self.shape//2))
        self.tildeMBI -= self.temp_gain * self.strength * shift(self.tildeMBB, self.index-self.shape//2)
        self.BI -= self.temp_gain * self.strength * shift(self.BB, self.index-self.shape//2)

        self.clean_map += self.temp_gain * self.strength * shift(self.psf, self.index-self.shape//2)

    def _update_basis_function(self):
        dmap = self.tildeII - self.mod
        
        gain = 0.1
        
        index = np.asarray( np.unravel_index( self.indices_corr[ np.argmax(np.abs(dmap).flatten()[self.window_corr]) ], self.shape) )
        strength = dmap[index[0], index[1]]
        dmap -= gain * strength * shift(self.BB, index-self.shape//2)
        self.basis_x.append([index, strength, gain])
        
        self.cmap += gain*strength*shift(self.delta, index-self.shape//2)
        self.mod += gain*strength*shift(self.BB, index-self.shape//2)
        
        #Now we add the conjugate as well
        index_conj = self.shape//2-(index-self.shape//2)
        self.basis_x.append([index_conj, strength, gain])
        
        self.cmap += gain*strength*shift(self.delta, index_conj-self.shape//2)
        self.mod += gain*strength*shift(self.BB, index_conj-self.shape//2)
        
        return self.cmap.copy()

























