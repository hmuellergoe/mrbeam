import numpy as np
from scipy.signal import fftconvolve

from regpy.operators import Operator
from regpy import discrs, util
from regpy.solvers import Solver
from regpy.stoprules import StopRule
from regpy.discrs import DirectSum

class FourierTransform2D(Operator):
    def __init__(self, domain, padding=0):
        assert isinstance(domain, discrs.UniformGrid)
        assert 2*(padding//2) == padding
        self.N = domain.shape[0]+padding
        self.padding = padding
        codomain_spacing = 1/(self.N*domain.spacing)
        codomain = discrs.UniformGrid((-0.5*codomain_spacing[0]*self.N, 0.5*codomain_spacing[0]*(self.N-2), self.N),
                                    (-0.5*codomain_spacing[1]*self.N, 0.5*codomain_spacing[1]*(self.N-2), self.N), dtype=complex)
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        x = np.pad(x, self.padding//2)
        x = np.fft.ifftshift(x)
        y = np.fft.fftn(x, norm='ortho')
        return np.fft.fftshift(y)

    def _adjoint(self, y):
        y = np.fft.ifftshift(y)
        x = np.fft.ifftn(y, norm='ortho')
        x = np.fft.fftshift(x)[self.padding//2:self.N-self.padding//2, self.padding//2:self.N-self.padding//2]
        if self.domain.is_complex:
            return x
        else:
            return np.real(x)

    @property
    def inverse(self):
        return self.adjoint

    def __repr__(self):
        return util.make_repr(self, self.domain)
    
class Convolution(Operator):
    def __init__(self, domain, beam):
#        assert isinstance(domain, discrs.UniformGrid)
        codomain = domain 
        super().__init__(domain, codomain, linear=True)
        assert np.isreal(beam).any()
        self.beam = beam
        
    def _eval(self, x):
        return fftconvolve(self.beam, x, mode='same')
        
    def _adjoint(self, y):
        return fftconvolve(self.beam, y, mode='same')
    
class Reshape(Operator):
    def __init__(self, domain, codomain):
        assert np.prod(domain.shape) == np.prod(codomain.shape)
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        return x.reshape(self.codomain.shape)
    
    def _adjoint(self, y):
        return y.reshape(self.domain.shape)

    @property
    def inverse(self):
        return self.adjoint

    def __repr__(self):
        return util.make_repr(self, self.domain)
    
class RegpySolver(Solver):
    def __init__(self, callback=None):
        super().__init__()
        self.__converged = False
        self.callback = callback
        
    def next(self):
        """Perform a single iteration.

        Returns
        -------
        boolean
            False if the solver already converged and no step was performed.
            True otherwise.
        """
        if self.__converged:
            return False
        self._next()
        if self.callback != None:
            self.callback(self.x)
        return True
    
class Display(StopRule):
    def __init__(self, functional, string):
        super().__init__()
        self.functional = functional
        self.string = string
    
    def __repr__(self):
        return 'Display'

    def _stop(self, x, y=None):
        self.log.info(self.string + '--> {}'.format( self.functional(x) ))
        return False
    
def gradientuniformgrid(u, spacing=1):
    """Computes the gradient of field given by 'u'. 'u' is defined on a 
    equidistant grid. Returns a list of vectors that are the derivatives in each 
    dimension."""
    return 1/spacing*np.array(np.gradient(u))

def divergenceuniformgrid(u, dim, spacing=1):
    """Computes the divergence of a vector field 'u'. 'u' is assumed to be
    a list of matrices u=(u_x, u_y, u_z, ...) holding the values for u on a
    regular grid"""
    return 1/spacing*np.ufunc.reduce(np.add, [np.gradient(u[i], axis=i) for i in range(dim)])

def power(space, exponent):
    assert isinstance(exponent, int)
    domain = space
    for i in range(exponent-1):
        domain = DirectSum(domain, space, flatten=True)
    return domain