from lightwise import imgutils, wtutils, nputils

from MSI.utils.beams import Bessel2D as bessel

import numpy as np

from scipy.signal import fftconvolve

'''
Possible Parameters:
wavelet_fct: 'b1', 'b3', 'b5', 'db1'-'db18', 'sym2'-'sym8', 'coif1'-'coif5', 'triangle', 'triangle2' 
wt_dec: uiwt, uimwt, dwt, uwt
'''    

class WTDictionary():    
    def __init__(self, wavelet_fct='b1', wt_dec = 'uiwt', min_scale=1, max_scale=4):
        self.wavelet_fct = wavelet_fct
        if wt_dec == 'uiwt':
            self.wt_dec = wtutils.uiwt
        elif wt_dec == 'uiwmt':
            self.wt_dec = wtutils.uiwmt            
        elif wt_dec == 'dwt':
            self.wt_dec = wtutils.dwt            
        elif wt_dec == 'uwt':
            self.wt_dec = wtutils.uwt
        else:
            raise NotImplementedError()
        self.min_scale = min_scale
        self.max_scale = max_scale  
        
        if self.wavelet_fct in ['b3', 'triangle2']:
            self.scales_width = [max(1.5, 3 * min(1, j) * pow(2, max(0, j - 1))) for j in range(self.min_scale, self.max_scale)]
        else:
            self.scales_width = [max(1, 2 * min(1, j) * pow(2, max(0, j - 1))) for j in range(self.min_scale, self.max_scale)]

        self.length = len(self.scales_width)

    def compute_scaling_functions(self):
        list_scaling_functions = []
        assert self.scales_width[0] == 1        
        cj = np.array([1])
        list_scaling_functions.append(cj)
        for i in range(self.length):
            level = self.scales_width[i]
            hkd = nputils.atrou(wtutils.get_wavelet_obj(self.wavelet_fct).get_dec_hk(), level)
            cj = fftconvolve(cj, hkd, mode='full')
            list_scaling_functions.append(cj)
        return list_scaling_functions

    def compute_wavelets(self):
        list_scaling_functions = self.compute_scaling_functions()
        list_wavelets = []
        for item in nputils.nwise(list_scaling_functions, 2):
            to_pad = ( len(item[1])-len(item[0]) ) // 2
            list_wavelets.append( np.pad(item[0], to_pad) - item[1] )
        list_wavelets.append(list_scaling_functions[-1])
        return list_wavelets
    
class DOGDictionary():
    def __init__(self, widths, shape, angle=0, ellipticities=1, nsigma=None, support=None, **args):
        self.widths = widths
        self.angle = angle
        self.ellipticity = ellipticities
        self.shape = shape
        
        self.nsigma=nsigma
        if nsigma==None:
            self.nsigma = 50
        
        self.length = len(self.widths)#+1
        
        self.dirac = np.zeros(self.shape)
        self.dirac[self.shape[0]//2, self.shape[1]//2] = 1      
        
    #Computes scaling functions as imgutils.GaussianBeam object and not as numpy,
    #thus only useful for inner computation in the MSI module
    def _compute_scaling_functions(self):
        beams = [imgutils.GaussianBeam(self.ellipticity * w, w, bpa=self.angle, nsigma=self.nsigma) for w in self.widths]
        return beams
    
    def compute_dogs(self):
        beams = self._compute_scaling_functions()
        filtered = [b.convolve(self.dirac) for b in beams]
        list_wavelets = [(el[0] - el[-1]) for el in nputils.nwise(filtered, 2)]
        list_wavelets.append( filtered[-1] )
#        delta_wavelet = [filtered[0]]
#        list_wavelets = np.concatenate((delta_wavelet, list_wavelets))
        return list_wavelets

    def set_smoothing_scale(self, list_wavelets, width):
        beam = imgutils.GaussianBeam(self.ellipticity * width, width, bpa=self.angle, nsigma=self.nsigma)
        list_wavelets[-1]=beam.convolve(self.dirac)
        return list_wavelets

class BesselDictionary():
    def __init__(self, widths, shape, nsigma=10, support=None, **args):
        self.widths = widths
        self.shape = shape
        self.nsigma = nsigma
        self.support = support
        
        self.length = len(self.widths)
        
        self.dirac = np.zeros(self.shape)
        self.dirac[self.shape[0]//2, self.shape[1]//2] = 1      
        
    #Computes scaling functions as imgutils.GaussianBeam object and not as numpy,
    #thus only useful for inner computation in the MSI module
    def _compute_scaling_functions(self):
        beams = [bessel(w, w, nsigma=self.nsigma, support=self.support) for w in self.widths]
        return beams
    
    def compute_wavelets(self):
        beams = self._compute_scaling_functions()
        filtered = [b.convolve(self.dirac) for b in beams]
        list_wavelets = [(el[0] - el[-1]) for el in nputils.nwise(filtered, 2)]
        list_wavelets.append( filtered[-1] )
        return list_wavelets

    def set_smoothing_scale(self, list_wavelets, width):
        del(list_wavelets[-1])
        beam = bessel(width, width, nsigma=self.nsigma, support=self.support)
        list_wavelets.append(beam.convolve(self.dirac))
        return list_wavelets

class RectDictionary():
    def __init__(self, widths, grid, dist=False, **args):
        self.widths = widths
        self.grid = grid

        if dist:
            self.diff = self.grid.coords
            
        else:    
            self.index = np.asarray(self.grid.shape) // 2
            self.center = self.grid.coords[:, self.index[0], self.index[1]]
    
            self.diff = np.sqrt( (self.grid.coords[0]-self.center[0])**2 
                + (self.grid.coords[1]-self.center[1])**2 )

        self.length = len(self.widths)
        
    def _compute_scaling_functions(self):
        beams = [self._rect(w) for w in self.widths]
        return beams

    def compute_rects(self):
        beams = self._compute_scaling_functions()
        list_wavelets = [(el[0] - el[-1]) for el in nputils.nwise(beams, 2)]
        list_wavelets.append( beams[-1] )
        return list_wavelets
    
    def set_smoothing_scale(self, list_rects, width):
        del(list_rects[-1])
        beam = self._rect(width)
        list_rects.append(beam)
        return list_rects

    def _rect(self, width):
        return 1-np.heaviside(self.diff * width-1/2, 0)

