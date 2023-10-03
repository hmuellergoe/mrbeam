"""Finite element discretizations using NGSolve

This module implements a `regpy.discrs.Discretization` instance for NGSolve spaces and corresponding
Hilbert space structures. Operators are in the `regpy.operators.ngsolve` module.
"""

import ngsolve as ngs
import numpy as np

from regpy.discrs import Discretization, DirectSum
from regpy.hilbert import HilbertSpace, L2, L2Boundary, Sobolev, SobolevBoundary
from regpy.operators import Operator
from regpy.util import memoized_property


class NgsSpace(Discretization):
    """A discretization wrapping an `ngsolve.FESpace`.

    Parameters
    ----------
    fes : ngsolve.FESpace
       The wrapped NGSolve discretization.
    """

    def __init__(self, fes, bdr=None):
        assert isinstance(fes, ngs.FESpace)
        super().__init__(fes.ndof)
        self.fes = fes
        self.bdr = bdr
        self._fes_util = ngs.L2(fes.mesh, order=0)
        self._gfu_util = ngs.GridFunction(self._fes_util)
        self._gfu_fes = ngs.GridFunction(fes)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.fes == other.fes

    def ones(self):
        self._gfu_fes.Set(1)
        return self._gfu_fes.FV().NumPy().copy()

    def rand(self, rand=np.random.random_sample):
        r = rand(self._fes_util.ndof)
        self._gfu_util.vec.FV().NumPy()[:] = r
        self._gfu_fes.Set(self._gfu_util)
        return self._gfu_fes.vec.FV().NumPy().copy()
    
    def randn(self):
        return self.rand(np.random.standard_normal)

    def to_ngs(self, array):
        gf = ngs.GridFunction(self.fes)
        gf.vec.FV().NumPy()[:] = array
        return gf

    def from_ngs(self, coeff):
        self._gfu_fes.Set(coeff)
        return self._gfu_fes.vec.FV().NumPy().copy()
    
    def draw(self, coefficient_array, name):
        assert isinstance(name, str)
        gfu_fes = ngs.GridFunction(self.fes)
        gfu_fes.vec.FV().NumPy()[:] = coefficient_array
        coefficientfunction = ngs.CoefficientFunction( gfu_fes )
        ngs.Draw(coefficientfunction, self.fes.mesh, name)

    def __add__(self, other):
        if isinstance(other, Discretization):
            product_space = DirectSum(self, other, flatten=True)
            if all(product_space.summands[i]==product_space.summands[0] for i in range(len(product_space.summands))):
                product_space.fes = product_space.summands[0].fes
                product_space.bdr = product_space.summands[0].bdr
            return product_space
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Discretization):
            product_space = DirectSum(other, self, flatten=True)
            if all(product_space.summands[i]==product_space.summands[0] for i in range(len(product_space.summands))):
                product_space.fes = product_space.summands[0].fes
                product_space.bdr = product_space.summands[0].bdr
            return product_space            
        else:
            return NotImplemented

    def __pow__(self, power):
        assert isinstance(power, int)
        domain = self
        for i in range(power-1):
            domain = DirectSum(domain, self, flatten=True)
        domain.fes = self.fes
        domain.bdr = self.bdr
        return domain


    
class Matrix(Operator):
    """An operator defined by an NGSolve bilinear form.

    Parameters
    ----------
    domain : NgsSpace
        The discretization.
    form : ngsolve.BilinearForm or ngsolve.BaseMatrix
        The bilinear form or matrix. A bilinear form will be assembled.
    """

    def __init__(self, domain, form):
        assert isinstance(domain, NgsSpace)
        if isinstance(form, ngs.BilinearForm):
            assert domain.fes == form.space
            form.Assemble()
            mat = form.mat
        elif isinstance(form, ngs.BaseMatrix):
            mat = form
        else:
            raise TypeError('Invalid type: {}'.format(type(form)))
        self.mat = mat
        """The assembled matrix."""
        super().__init__(domain, domain, linear=True)
        self._gfu_in = ngs.GridFunction(domain.fes)
        self._gfu_out = ngs.GridFunction(domain.fes)
        self._inverse = None

    def _eval(self, x):
        self._gfu_in.vec.FV().NumPy()[:] = x
        self._gfu_out.vec.data = self.mat * self._gfu_in.vec
        return self._gfu_out.vec.FV().NumPy().copy()

    def _adjoint(self, y):
        self._gfu_in.vec.FV().NumPy()[:] = y
        self._gfu_out.vec.data = self.mat.T * self._gfu_in.vec
        return self._gfu_out.vec.FV().NumPy().copy()

    @property
    def inverse(self):
        """The inverse as a `Matrix` instance."""
        if self._inverse is not None:
            return self._inverse
        else:
            self._inverse = Matrix(
                self.domain,
                self.mat.Inverse(freedofs=self.domain.fes.FreeDofs())
            )
            self._inverse._inverse = self
            return self._inverse


@L2.register(NgsSpace)
class L2FESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.L2` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v)
        return Matrix(self.discr, form)


@Sobolev.register(NgsSpace)
class SobolevFESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.Sobolev` on an `NgsSpace`."""
    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(u * v + ngs.grad(u) * ngs.grad(v))
        return Matrix(self.discr, form)


@L2Boundary.register(NgsSpace)
class L2BoundaryFESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.L2Boundary` on an `NgsSpace`."""
    def __init__(self, discr):
        assert discr.bdr is not None
        super().__init__(discr)

    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace(),
            definedon=self.discr.fes.mesh.Boundaries(self.discr.bdr)
        )
        return Matrix(self.discr, form)


@SobolevBoundary.register(NgsSpace)
class SobolevBoundaryFESpace(HilbertSpace):
    """The implementation of `regpy.hilbert.SobolevBoundary` on an `NgsSpace`."""
    def __init__(self, discr):
        assert discr.bdr is not None
        super().__init__(discr)


    @memoized_property
    def gram(self):
        u, v = self.discr.fes.TnT()
        form = ngs.BilinearForm(self.discr.fes, symmetric=True)
        form += ngs.SymbolicBFI(
            u.Trace() * v.Trace() + u.Trace().Deriv() * v.Trace().Deriv(),
            definedon=self.discr.fes.mesh.Boundaries(self.discr.bdr)
        )
        return Matrix(self.discr, form)

'''Special NGSolve functionals'''
from regpy.functionals import Functional
from regpy.functionals import L1, TV
@L1.register(NgsSpace)
class NgsL1(Functional):
    def __init__(self, domain):
        self._gfu = ngs.GridFunction(domain.fes)
        self._fes_util = ngs.L2(domain.fes.mesh, order=0)
        self._gfu_util = ngs.GridFunction(self._fes_util)
        super().__init__(domain)

    def _eval(self, x):
        self._gfu.vec.FV().NumPy()[:] = x
        coeff = ngs.CoefficientFunction(self._gfu)
        return ngs.Integrate( ngs.Norm(coeff), self.domain.fes.mesh )

    def _gradient(self, x):
        self._gfu.FV().NumPy()[:] = x
        self._gfu_util.Set(self._gfu)
        y = self._gfu_util.vec.FV().NumPy()
        self._gfu_util.vec.FV().NumPy()[:] = np.sign(y)
        self._gfu.Set(self._gfu_util)
        return self._gfu.vec.FV().NumPy().copy()

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau): 
        self._gfu.vec.FV().NumPy()[:] = x
        self._gfu_util.Set(self._gfu)
        y = self._gfu_util.vec.FV().NumPy()
        self._gfu_util.vec.FV().NumPy()[:] = np.maximum(0, np.abs(y)-tau)*np.sign(y)
        self._gfu.Set(self._gfu_util)
        return self._gfu.vec.FV().NumPy().copy()

@TV.register(NgsSpace)
class NgsTV(Functional):
    def __init__(self, domain):
        super().__init__(domain)
        self._gfu = ngs.GridFunction(self.domain.fes)
        self._gfu.Set(0)
        self._p = list(ngs.grad(self._gfu))
        self._q = list(ngs.grad(self._gfu))
        self._gfu_div = ngs.GridFunction(domain.fes)
        self._gfu_div.vec.FV().NumPy()[:] = ngsdivergence(self._p, self.domain.fes)
        self._fes_util = ngs.L2(self.domain.fes.mesh, order=0)
        self._gfu_util = ngs.GridFunction(self._fes_util)

    def _eval(self, x):
        self._gfu.vec.FV().NumPy()[:] = x
        gradu = ngs.grad(self._gfu)
        tvnorm = 0
        for i in range(gradu.dim):
            self._gfu_util.Set(gradu[i])
            tvnorm += ngs.Integrate( ngs.Norm(self._gfu_util), self.domain.fes.mesh )
        return tvnorm

    def _gradient(self, x):
        raise NotImplementedError

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau, stepsize=0.1, maxiter=10):
        self._gfu.Set(0)
        self._p = list(ngs.grad(self._gfu))

        self._gfu.vec.FV().NumPy()[:] = x
        self._gfu_update = ngs.GridFunction(self.domain.fes)
        self._gfu_out = ngs.GridFunction(self.domain.fes)
        for i in range(maxiter):
            self._gfu_update.Set( self._gfu_div - self._gfu/tau )
            update= stepsize * ngs.grad( self._gfu_update )
            #Calculate |update|
            for i in range(len(self._p)):
                self._q[i] = 1+ngs.Norm(update[i])
                self._p[i] = (self._p[i] + update[i]) / self._q[i]
            self._gfu_div.vec.FV().NumPy()[:] = ngsdivergence(self._p, self.domain.fes)
        self._gfu_out.Set(self._gfu - tau*self._gfu_div)
        return self._gfu_out.vec.FV().NumPy().copy()        

"""Computes the divergence of a vector field 'p' on a FES 'fes'. gradp is a list of ngsolve CoefficientFunctions
    p=(p_x, p_y, p_z, ...). The return value is the coefficient array of the GridFunction holding the divergence."""

def ngsdivergence(p, fes):
    toret = np.zeros(fes.ndof)
    gfu_in = ngs.GridFunction(fes)
    gfu_out = ngs.GridFunction(fes)
    for i in range(len(p)):
        gfu_in.Set(p[i])
        coeff = ngs.grad(gfu_in)[i]
        gfu_out.Set(coeff)
        toret += gfu_out.vec.FV().NumPy().copy()
    return toret
