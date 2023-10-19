import numpy as np

from regpy.operators import Operator

from MSI.MSDecomposition import WaveletTransform2D as msiWaveletTransform2D
from MSI.MSDecomposition import DoG2D as msiDoG2D
from MSI.MSMDDecomposition import DoG2D as msmdDoG2D
from MSI.MSDictionaries import WTDictionary as msiWTDictionary
from MSI.MSDictionaries import DOGDictionary as msiDOGDictionary
from MSI.MSMDDictionaries import DOGDictionary as msmdDOGDictionary
from MSI.MSDictionaries import RectDictionary as msiRectDictionary
from MSI.MSDictionaries import BesselDictionary as msiBesselDictionary
from MSI.MSMDDictionaries import BesselDictionary as msmdBesselDictionary

from lightwise.nputils import convolve

from joblib import Parallel, delayed

class WaveletTransform(Operator):
    def __init__(self, domain, wavelet_fct='b1', wt_dec = 'uiwt', min_scale=1, max_scale=4, **args):
        assert len(domain.shape) == 2
        
        self.wt_trafo = msiWaveletTransform2D(wavelet_fct=wavelet_fct, wt_dec = wt_dec,
                                               min_scale=min_scale, max_scale=max_scale)
        
        self.shape = domain.shape
        self.nr_scales = int(max_scale-min_scale)
        self.toret = np.zeros((self.nr_scales, self.shape[0], self.shape[1]))
        
        codomain = domain**self.nr_scales
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        list_scales = self.wt_trafo.decompose( x )
        self._extract(list_scales)
        return self.toret.flatten()
    
    def _adjoint(self, y):
        scales = y.reshape(self.nr_scales, self.shape[0], self.shape[1])
        for i in range(self.nr_scales):
            self.toret[i] = self.wt_trafo.compute_scale(scales[i], i)
        return np.sum(self.toret, axis=0)
        
    def _extract(self, list_scales):
        for i in range(self.nr_scales):
            self.toret[i] = list_scales[i][0]


class DOGTransform(Operator):
    def __init__(self, domain, widths, angle=0, ellipticities=1, md=False, **args):
        assert len(domain.shape) == 2
        
        if md:
            self.dog_trafo = msmdDoG2D(widths, angle=angle, ellipticities=ellipticities)
        else:
            self.dog_trafo = msiDoG2D(widths, angle=angle, ellipticities=ellipticities)
        
        self.shape = domain.shape
        self.nr_scales = len(widths)
        self.toret = np.zeros((self.nr_scales, self.shape[0], self.shape[1]))
        
        codomain = domain**self.nr_scales
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        list_scales = self.dog_trafo.decompose( x )
        self._extract(list_scales)
        self.toret[-1] = x - np.sum(self.toret[:-1, :], axis=0)
        return self.toret.flatten()
    
    def _adjoint(self, y):
        scales = y.reshape(self.nr_scales, self.shape[0], self.shape[1])
        last_scale = scales[-1]
        scales = scales[:-1]-last_scale
        for i in range(self.nr_scales-1):
            self.toret[i] = self.dog_trafo.compute_scale(scales[i], i)
        return np.sum(self.toret, axis=0)+last_scale
        
    def _extract(self, list_scales):
        for i in range(self.nr_scales-1):
            self.toret[i] = list_scales[i][0]       


class DWTDictionary(Operator):
    def __init__(self, domain, codomain, wavelet_fct='b1', wt_dec = 'uiwt', min_scale=1, max_scale=4, shape=None, **args):
        self.wtdict = msiWTDictionary(wavelet_fct=wavelet_fct, wt_dec=wt_dec, min_scale=min_scale, max_scale=max_scale)
        self.wavelets = self.wtdict.compute_wavelets()

        self.length = self.wtdict.length+1

#        for i in range(self.length):
#            self.wavelets[i] /= np.max(self.wavelets[i])

        super().__init__(domain, codomain, linear=True)

        if shape == None:
            shape = codomain.shape
        self.shape = shape

        self.to_pad = (self.codomain.shape[0]-self.shape[0])//2

    def _eval(self, x):
        array = x.reshape((self.length, self.shape[0], self.shape[1]))
        toret = np.zeros(self.codomain.shape)
        for i in range(self.length):
            to_convolve = np.pad(array[i], self.to_pad)
            toret += convolve(to_convolve, self.wavelets[i], mode='same')
        return toret

    def _adjoint(self, y):
        toret = np.zeros((self.length, self.codomain.shape[0], self.codomain.shape[1]))
        for i in range(self.length):
            toret[i, :, :] = convolve(y, self.wavelets[i], mode='same')
        return toret[ : ,  self.to_pad:self.to_pad+self.shape[0] , self.to_pad:self.to_pad+self.shape[1] ].flatten()
    
class DOGDictionary(Operator):
    def __init__(self, domain, codomain, widths, shape, ellipticities = 1, angle = 0, md=False, num_cores=1, smoothing_scale=None, **args):
        self.num_cores = num_cores
        
        if md:
            self.dogdict = msmdDOGDictionary(widths, shape, ellipticities=ellipticities, angle=angle, **args)
        else:
            self.dogdict = msiDOGDictionary(widths, shape, ellipticities=ellipticities, angle=angle, **args)
        self.dogs = self.dogdict.compute_dogs()
        
        if smoothing_scale != None:
            self.set_smoothing_scale(smoothing_scale)
        
        self.length = self.dogdict.length
        self.shape = shape

        self.weights = np.ones(self.length)
        
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        array = x.reshape((self.length, self.shape[0], self.shape[1]))
        toret = np.zeros(self.codomain.shape)
        res = Parallel(n_jobs=self.num_cores)(delayed(convolve)(array[i], self.dogs[i], mode='same') for i in range(self.length))
        for i in range(self.length):
            toret += self.weights[i] * res[i]
        return toret
    
    def _adjoint(self, y):
        toret = np.zeros((self.length, self.shape[0], self.shape[1]))
        res = Parallel(n_jobs=self.num_cores)(delayed(convolve)(y, self.dogs[i], mode='same') for i in range(self.length))
        for i in range(self.length):
            toret[i, :, :] = self.weights[i] * res[i]
        return toret.flatten()
    
    def set_weights(self, weights):
        self.weights = weights

    def set_smoothing_scale(self, width):
        self.dogs = self.dogdict.set_smoothing_scale(self.dogs.copy(), width)

class RectDictionary(Operator):
    def __init__(self, domain, widths, fourier, shape, **args):
        self.fourier = fourier

        self.rectdict = msiRectDictionary(widths, fourier.codomain, **args)
        self.rects = self.rectdict.compute_rects()

        self.length = self.rectdict.length
        self.shape = shape
        
        self.weights = np.ones(self.length)

        super().__init__(domain, fourier.codomain, linear=True)

    def _eval(self, x):
        array = x.reshape((self.length, self.shape[0], self.shape[1]))
        toret = np.zeros(self.codomain.shape, dtype=complex)
        for i in range(self.length):
            toret += self.weights[i] * self.fourier(array[i]) * self.rects[i]
        return toret

    def _adjoint(self, y):
        toret = np.zeros((self.length, self.shape[0], self.shape[1]))
        for i in range(self.length):
            toret[i, :, :] = self.weights[i] * self.fourier.adjoint( self.rects[i] * y )
        return toret.flatten()
    
    def set_weights(self, weights):
        self.weights = weights
        
    def set_smoothing_scale(self, width):
        self.rects = self.rectdict.set_smoothing_scale(self.rects.copy(), width)

class BesselDictionary(Operator):
    def __init__(self, domain, codomain, widths, shape, md=False, num_cores=1, smoothing_scale=None, **args):
        self.num_cores = num_cores

        if md:
            self.besseldict = msmdBesselDictionary(widths, shape, **args)
        else:
            self.besseldict = msiBesselDictionary(widths, shape, **args)
        self.wavelets = self.besseldict.compute_wavelets()
        
        if smoothing_scale != None:
            self.set_smoothing_scale(smoothing_scale)

        self.length = self.besseldict.length
        self.shape = shape
        
        self.weights = np.ones(self.length)

        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        array = x.reshape((self.length, self.shape[0], self.shape[1]))
        toret = np.zeros(self.codomain.shape)
        res = Parallel(n_jobs=self.num_cores)(delayed(convolve)(array[i], self.wavelets[i], mode='same') for i in range(self.length))
        for i in range(self.length):
            toret += self.weights[i] * res[i]
        return toret
    
    def _adjoint(self, y):
        toret = np.zeros((self.length, self.shape[0], self.shape[1]))
        res = Parallel(n_jobs=self.num_cores)(delayed(convolve)(y, self.wavelets[i], mode='same') for i in range(self.length))
        for i in range(self.length):
            toret[i, :, :] = self.weights[i] * res[i]
        return toret.flatten()
    
    def set_weights(self, weights):
        self.weights = weights

    def set_smoothing_scale(self, width):
        self.wavelets = self.besseldict.set_smoothing_scale(self.wavelets.copy(), width)