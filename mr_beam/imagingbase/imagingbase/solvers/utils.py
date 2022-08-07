from scipy.optimize import curve_fit   
from imagingbase.operators.msi import DOGDictionary 
from imagingbase.regpy_utils import power
from regpy.discrs import Discretization
import numpy as np

from copy import deepcopy
  
class BuildMerger():
    def __init__(self, obj):
        self.obj = deepcopy(obj)
      
    def _build_merger(self):
        widths = self._match_bessel_dog()
        grid = Discretization(self.obj.dmap.shape)
        domain = power(grid, self.obj.merger.length)
        merger = DOGDictionary(domain, grid, widths, grid.shape, num_cores=self.obj.wrapper.num_cores, **self.obj.args)
        return merger
        
    def _match_bessel_dog(self):
        beams = self.obj.merger.besseldict._compute_scaling_functions()
        if self.obj.md:
            beams = beams[1]
        filtered = [b.convolve(self.obj.merger.besseldict.dirac) for b in beams]
        bessels = np.asarray(filtered)
        
        size = np.asarray([bessels.shape[1], bessels.shape[2]])
        
        x, y = np.indices(size, dtype=int)
        center = np.floor(size / 2.)
    
        xp = (x-center[0])
        yp = (y-center[1])
        
        diff = np.sqrt(xp**2+yp**2)
        
        dog_widths = np.zeros(bessels.shape[0])
        
        def gaussian(diff, sigma):
            toret = 1/(2*np.pi*sigma**2) * np.exp(-diff**2/(2*sigma**2))
            return toret.flatten()
        
        for i in range(len(dog_widths)):
            popt, _ = curve_fit(gaussian, diff, bessels[i].flatten())
            dog_widths[i] = popt
            
        return 2.355*dog_widths
    
def delta(shape):
    toret = np.zeros(shape)
    toret[tuple(np.asarray(shape)//2)] = 1
    return toret
    
import numba
from numba import njit

#@njit()
#def shift1D(arr, num):
#    if num > 0:
#        return np.concatenate((np.full(num, 0), arr[:-num]))
#    else:
#        return np.concatenate((arr[-num:], np.full(-num, 0))) 
    
#@njit(parallel=True)    
#def shift2D(argument, shifting):
#    arr = argument.copy()
#    for i in range(arr.shape[0]):
#        arr[i] = shift1D(arr[i], shifting[1])
#    for i in range(arr.shape[0]):
#        arr[:,i] = shift1D(arr[:,i], shifting[0])
#    return arr

def _shift1D_2D(arr, num):
    if num > 0:
        return np.concatenate((np.full((arr.shape[0], num), 0), arr[:,:-num]), axis=1)
    else:
        return np.concatenate((arr[:,-num:], np.full((arr.shape[0],-num), 0)), axis=1) 

def shift2D(argument, shifting):
    arr = argument.copy()
    arr = _shift1D_2D(arr, shifting[1])
    arr = _shift1D_2D(arr.transpose(), shifting[0])
    return arr.transpose()

#def _shift1D_3D(arr, num):
#    if num > 0:
#        return np.concatenate((np.full((arr.shape[0], arr.shape[1], num), 0), arr[:,:,:-num]), axis=2)
#    else:
#        return np.concatenate((arr[:,:,-num:], np.full((arr.shape[0], arr.shape[1], -num), 0)), axis=2)     

#def shift3D(argument, shifting):
#    arr = argument.copy()
#    arr = _shift1D_3D(arr, shifting[1])
#    arr = _shift1D_3D(arr.transpose((0, 2, 1)), shifting[0])
#    return arr.transpose((0, 2, 1))