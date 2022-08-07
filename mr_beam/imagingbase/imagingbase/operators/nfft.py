import numpy as np

#When working on windows: Import pynfft inside mingw by specifying the path
#to the pynfft directory
import sys
try:
    sys.path.append('C:\\msys64\\home\\pyNFFT-master')
except:
    print('Warning: pynfft directory not found')
    print('Can be ignored when working on linux or macos')
import pynfft

from regpy.operators import Operator
from regpy.discrs import Discretization, UniformGrid
from regpy.util import memoized_property


class NFFT(Operator):
    def __init__(self, grid, nodes, weights):
        assert isinstance(grid, UniformGrid)
        assert nodes.shape[1] == grid.ndim

        # pynfft computes a sum of the form
        #      sum_k f_k exp(-2 pi i k x)
        # where k ranges from -n/2 to n/2-1. Our frequencies are actually defined
        # `grid`, so we would like
        #      sum_k f_k exp(-i k x)
        # where k ranges over `grid`. This is equivalent to rescaling the nodes x
        # by the following factor (also handling the multidimensional case):
        scaling_factor = 1 * grid.extents / np.asarray(grid.shape)
        nodes = scaling_factor * nodes
        # The nodes' inversion weights need to be scaled accordingly.
        self.weights = np.prod(scaling_factor) * weights

        super().__init__(
            domain=grid,
            codomain=Discretization(nodes.shape[0], dtype=complex),
            linear=True
        )

        # Initialize the NFFT
        self.plan = pynfft.NFFT(N=grid.shape, M=nodes.shape[0])
        self.plan.x = nodes
        self.plan.precompute()

        # NFFT quadrature factor
        self.nfft_factor = grid.volume_elem

        # Initialize the Solver for computing the inverse NFFT
        # TODO unused?
        self.solver = pynfft.Solver(self.plan)
        self.solver.w = self.weights

    def _eval(self, x):
        self.plan.f_hat = x
        return self.nfft_factor * self.plan.trafo()

    def _adjoint(self, y):
        self.plan.f = y
        x = self.nfft_factor * self.plan.adjoint()
        if self.domain.is_complex:
            return x
        else:
            return np.real(x)

    @memoized_property
    def inverse(self):
        # TODO add solver-based inverse
        return ApproxInverseNFFT(self)


class ApproxInverseNFFT(Operator):
    def __init__(self, op):
        self.op = op
        super().__init__(
            domain=op.codomain,
            codomain=op.domain,
            linear=True
        )

    def _eval(self, y):
        self.op.plan.f = self.op.weights * y
        x = self.op.plan.adjoint() / self.op.nfft_factor
        if self.op.domain.is_complex:
            return x
        else:
            return np.real(x)

    def _adjoint(self, x):
        self.op.plan.f_hat = x
        return np.conj(self.op.weights) * self.op.plan.trafo() / self.op.nfft_factor

class FFT(Operator):
    def __init__(self, grid, uvgrid):
        assert isinstance(grid, UniformGrid)
        assert isinstance(uvgrid, UniformGrid)
        assert uvgrid.ndim == grid.ndim

        super().__init__(
            domain=grid,
            codomain=uvgrid,
            linear=True
        )
#################################################################################################
        N_codomain = np.prod(uvgrid.shape)
        ndim = uvgrid.ndim
        nodes = uvgrid.coords.reshape((ndim, N_codomain)).transpose()
        scaling_factor = 1 * grid.extents / np.asarray(grid.shape)
        nodes = scaling_factor * nodes
        # The nodes' inversion weights need to be scaled accordingly.
        #self.weights = np.prod(scaling_factor) * weights

        # Initialize the NFFT
        self.plan = pynfft.NFFT(N=grid.shape, M=nodes.shape[0])
        self.plan.x = nodes
        self.plan.precompute()

        # NFFT quadrature factor
        self.nfft_factor = grid.volume_elem
#################################################################################################
        N_domain = np.prod(grid.shape)
        nodes_adjoint = grid.coords.reshape((ndim, N_domain)).transpose()
        scaling_factor = -1 * uvgrid.extents / np.asarray(uvgrid.shape)      
        nodes_adjoint =  scaling_factor * nodes_adjoint 

        self.plan_adjoint = pynfft.NFFT(N=uvgrid.shape, M=nodes_adjoint.shape[0])
        self.plan_adjoint.x = nodes_adjoint
        self.plan_adjoint.precompute()

        self.nfft_factor_adjoint = uvgrid.volume_elem

    def _eval(self, x):
        self.plan.f_hat = x
        return self.nfft_factor * self.plan.trafo().reshape(self.codomain.shape)

    def _adjoint(self, y):
        self.plan_adjoint.f_hat = y
        x = self.nfft_factor_adjoint * self.plan_adjoint.trafo()
        if self.domain.is_complex:
            return x.reshape(self.domain.shape)
        else:
            return np.real(x).reshape(self.domain.shape)

    @memoized_property
    def inverse(self):
        return self.adjoint