from collections import defaultdict

from copy import copy

import numpy as np

from regpy import operators, util, discrs, hilbert


class Functional:
    def __init__(self, domain):
        # TODO implement domain=None case
        assert isinstance(domain, discrs.Discretization)
        self.domain = domain
        self.Hdomain = hilbert.L2(domain)
        #Hdomain on which the proximal operator is evaluated
        #Overloaded if Hdomain != L2

    def __call__(self, x):
        assert x in self.domain
        try:
            y = self._eval(x)
        except NotImplementedError:
            y, _ = self._linearize(x)
        assert isinstance(y, float)
        return y

    def linearize(self, x):
        assert x in self.domain
        try:
            y, grad = self._linearize(x)
        except NotImplementedError:
            y = self._eval(x)
            grad = self._gradient(x)
        assert isinstance(y, float)
        assert grad in self.domain
        return y, grad

    def gradient(self, x):
        assert x in self.domain
        try:
            grad = self._gradient(x)
        except NotImplementedError:
            _, grad = self._linearize(x)
        assert grad in self.domain
        return grad

    def hessian(self, x):
        assert x in self.domain
        h = self._hessian(x)
        assert isinstance(h, operators.Operator)
        assert h.linear
        assert h.domain == h.codomain == self.domain
        return h

    def proximal(self, x, tau, proximal_pars = None):
        assert x in self.domain
        if proximal_pars == None:
            proximal_pars = {}
        self.proximal_pars = proximal_pars
        proximal = self._proximal(x, tau, **proximal_pars)
        assert proximal in self.domain
        return proximal

    def _eval(self, x):
        raise NotImplementedError

    def _linearize(self, x):
        raise NotImplementedError

    def _gradient(self, x):
        raise NotImplementedError

    def _hessian(self, x):
        return operators.ApproximateHessian(self, x)

    def _proximal(self, x, tau):
        return NotImplementedError

    def __mul__(self, other):
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return Composed(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return self * operators.Multiplication(self.domain, other)
        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            if other == 1:
                return self
            elif util.is_real_dtype(other):
                return LinearCombination((other, self))
        return NotImplemented

    def __truediv__(self, other):
        return (1 / other) * self

    def __add__(self, other):
        if isinstance(other, Functional):
            return LinearCombination(self, other)
        elif np.isscalar(other):
            return Shifted(self, other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
        return self


class Composed(Functional):
    def __init__(self, func, op):
        assert isinstance(func, Functional)
        assert isinstance(op, operators.Operator)
        assert func.domain == op.codomain
        super().__init__(op.domain)
        if isinstance(func, type(self)):
            op = func.op * op
            func = func.func
        self.func = func
        self.op = op

    def _eval(self, x):
        return self.func(self.op(x))

    def _linearize(self, x):
        y, deriv = self.op.linearize(x)
        z, grad = self.func.linearize(y)
        return z, deriv.adjoint(grad)

    def _gradient(self, x):
        y, deriv = self.op.linearize(x)
        return deriv.adjoint(self.func.gradient(y))

    def _hessian(self, x):
        if self.op.linear:
            return self.op.adjoint * self.func.hessian(x) * self.op
        else:
            # TODO this can be done slightly more efficiently
            return super()._hessian(x)

    def _proximal(self, x, tau):
        return NotImplementedError

#TODO: Add AbstractSum
class AbstractFunctionalBase:
    """Class representing abstract functionals without reference to a concrete implementation.

    Abstract functionals do not have elements, properties or any other structure, their sole purpose is
    to pick the proper concrete implementation for a given discretization.
    """

    def __add__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __rmul__(self, other):
        return NotImplemented


class AbstractFunctional(AbstractFunctionalBase):
    """An abstract functional that can be called on a discretization to get the corresponding
    concrete implementation.

    AbstractFunctionals provides two kinds of functionality:

    - A decorator method `register(discr_type)` that can be used to declare some class or function
      as the concrete implementation of this abstract functional for discretizations of type `discr_type`
      or subclasses thereof, e.g.:

              @TV.register(discrs.UniformGrid)
              class TVUniformGrid(HilbertSpace):
                  ...

    - AbstractFunctionals are callable. Calling them on a discretization and arbitrary optional
      keyword arguments finds the corresponding concrete `regpy.functionals.Functional` among all
      registered implementations. If there are implementations for multiple base classes of the
      discretization type, the most specific one will be chosen. The chosen implementation will
      then be called with the discretization and the keyword arguments, and the result will be
      returned.

      If called without a discretization as positional argument, it returns a new abstract functional
      with all passed keyword arguments remembered as defaults.

    Parameters
    ----------
    name : str
        A name for this abstract functional. Currently, this is only used in error messages, when no
        implementation was found for some discretization.
    """

    def __init__(self, name):
        self._registry = {}
        self.name = name
        self.args = {}

    def register(self, discr_type, impl=None):
        if impl is not None:
            self._registry.setdefault(discr_type, []).append(impl)
        else:
            def decorator(i):
                self.register(discr_type, i)
                return i
            return decorator

    def __call__(self, discr=None, **kwargs):
        if discr is None:
            clone = copy(self)
            clone.args = copy(self.args)
            clone.args.update(kwargs)
            return clone
        for cls in type(discr).mro():
            try:
                impls = self._registry[cls]
            except KeyError:
                continue
            kws = copy(self.args)
            kws.update(kwargs)
            for impl in impls:
                result = impl(discr, **kws)
                if result is NotImplemented:
                    continue
                assert isinstance(result, Functional)
                return result
        raise NotImplementedError(
            '{} not implemented on {}'.format(self.name, discr)
        )

L0 = AbstractFunctional('L0')
L1 = AbstractFunctional('L1')
TV = AbstractFunctional('TV')
HilbertNorm = AbstractFunctional('HilbertNorm')

class LinearCombination(Functional):
    def __init__(self, *args):
        coeff_for_func = defaultdict(lambda: 0)
        for arg in args:
            if isinstance(arg, tuple):
                coeff, func = arg
            else:
                coeff, func = 1, arg
            assert isinstance(func, Functional)
            assert np.isscalar(coeff) and util.is_real_dtype(coeff)
            if isinstance(func, type(self)):
                for c, f in zip(func.coeffs, func.funcs):
                    coeff_for_func[f] += coeff * c
            else:
                coeff_for_func[func] += coeff
        self.coeffs = []
        self.funcs = []
        for func, coeff in coeff_for_func.items():
            self.coeffs.append(coeff)
            self.funcs.append(func)

        domains = [op.domain for op in self.funcs if op.domain]
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains)
        else:
            domain = None

        super().__init__(domain)

    def _eval(self, x):
        y = 0
        for coeff, func in zip(self.coeffs, self.funcs):
            y += coeff * func(x)
        return y

    def _linearize(self, x):
        y = 0
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            f, g = func.linearize(x)
            y += coeff * f
            grad += coeff * g
        return y, grad

    def _gradient(self, x):
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            grad += coeff * func.gradient(x)
        return grad

    def _hessian(self, x):
        return operators.LinearCombination(
            *((coeff, func.hessian(x)) for coeff, func in zip(self.coeffs, self.funcs))
        )

    def _proximal(self, x, tau):
        return NotImplementedError

'''Helper to define Functionals with respective prox-operators on product spaces (discrs.DirectSum objects).
The functionals are given as a list of the functionals on the summands of the product space.'''
class FunctionalProductSpace(Functional):
    def __init__(self, funcs, domain, weights):
        assert isinstance(domain, discrs.DirectSum)
        self.length = len(domain.summands)
        for i in range(self.length):
            assert isinstance(funcs[i], Functional)
            assert funcs[i].domain == domain.summands[i] 
        self.funcs = funcs
        self.weights = weights
        super().__init__(domain)

    def _eval(self, x):
        splitted = self.domain.split(x)
        toret = 0 
        for i in range(self.length):
            toret += self.weights[i] * self.funcs[i](splitted[i])
        return toret

    def _gradient(self, x):
        splitted = self.domain.split(x)
        gradients = []
        for i in range(self.length):
            gradients.append( self.weights[i] * self.funcs[i].gradient(splitted[i]) )
        return np.asarray(gradients).flatten()

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, taus):
        assert len(taus) == self.length
        splitted = self.domain.split(x)
        proximals = []
        for i in range(self.length):
            proximals.append( self.funcs[i].proximal(splitted[i], self.weights[i]*taus[i]) )
        return np.asarray(proximals).flatten()

class Shifted(Functional):
    def __init__(self, func, offset):
        assert isinstance(func, Functional)
        assert np.isscalar(offset) and util.is_real_dtype(offset)
        super().__init__(func.domain)
        self.func = func
        self.offset = offset

    def _eval(self, x):
        return self.func(x) + self.offset

    def _linearize(self, x):
        return self.func.linearize(x)

    def _gradient(self, x):
        return self.func.gradient(x)

    def _hessian(self, x):
        return self.func.hessian(x)

    def _proximal(self, x, tau):
        return self.func.proximal(x, tau)

class Indicator(Functional):
    def __init__(self, domain, predicate):
        super().__init__(domain)
        self.predicate = predicate

    def _eval(self, x):
        if self.predicate(x):
            return 0
        else:
            return np.inf

    def _gradient(self, x):
        # This is of course not correct, but lets us use an Indicator functional to force
        # rejecting an MCMC proposals without altering the gradient.
        return self.domain.zeros()

    def _hessian(self, x):
        return operators.Zero(self.domain)

    """
    The proximal operator is the projection on the set predicate.
    However, it is more natural to implement indicator function constraints in Tikhonov 
    regularization by semismooth approaches. See semismooth Newton method.
    """
    def _proximal(self, x, tau):
        return NotImplementedError


class ErrorToInfinity(Functional):
    def __init__(self, func):
        super().__init__(func.domain)
        self.func = func

    def _eval(self, x):
        try:
            return self.func(x)
        except:
            return np.inf

    def _gradient(self, x):
        try:
            return self.func.gradient(x)
        except:
            return self.domain.zeros()

'''Generic implementation of the HilbertNorm 1/2*||x||**2. Proximal operator defined on hspace.'''
class HilbertNormGeneric(Functional):
    def __init__(self, hspace, Hdomain=None):
        assert isinstance(hspace, hilbert.HilbertSpace)
        super().__init__(hspace.discr)
        self.hspace = hspace
        self.Hdomain = Hdomain or hspace 
        '''overloads self.Hdomain from constructor'''

    def _eval(self, x):
        return np.real(np.vdot(x, self.hspace.gram(x))) / 2

    def _linearize(self, x):
        gx = self.hspace.gram(x)
        y = np.real(np.vdot(x, gx)) / 2
        return y, gx

    def _gradient(self, x):
        return self.hspace.gram(x)

    def _hessian(self, x):
        return self.hspace.gram

    def _proximal(self, x, tau, cgpars=None):
        if self.Hdomain == self.hspace:
            return 1/(1+tau)*x
        else:
            op = self.Hdomain.gram+tau*self.hspace.gram
            inverse = operators.CholeskyInverse(op)
            return inverse(self.Hdomain.gram(x))

'''Generic L1 Functional. Proximal implemented for default L2 hspace'''
class L0Generic(Functional):
    def __init__(self, domain):
        super().__init__(domain)

    def _eval(self, x):
        return float(np.count_nonzero(x))

    def _gradient(self, x):
        return NotImplementedError

    def _hessian(self, x):
        return NotImplementedError

    def _proximal(self, x, tau):
        toret = x.copy()
        toret = np.where(np.abs(x) <= np.sqrt(2 * tau), 0, toret)
        return toret

'''Generic L1 Functional. Proximal implemented for default L2 hspace'''
class L1Generic(Functional):
    def __init__(self, domain, weights=None):
        self.weights = weights or domain.ones()
        super().__init__(domain)

    def _eval(self, x):
        return np.sum(np.abs(self.weights * x))

    def _gradient(self, x):
        return self.weights * np.sign(x)

    def _hessian(self, x):
        # Even approximate Hessians don't work here.
        raise NotImplementedError

    def _proximal(self, x, tau):
        return np.maximum(0, np.abs(x)-self.weights*tau)*np.sign(x)

'''Generic TV Functional. Proximal implemented for default L2 hspace'''
class TVGeneric(Functional):
    def __init__(self, domain):
        super().__init__(domain)

    def _gradient(self, x):
        return NotImplementedError

    def _hessian(self, x):
        return NotImplementedError
    
    def _proximal(self, x, tau):
        return NotImplementedError

'''
Total Variation Norm: For C^1 functions the l1-norm of the gradient on a Uniform Grid
'''
from regpy.util import gradientuniformgrid
from regpy.util import divergenceuniformgrid
class TVUniformGrid(Functional):
    def __init__(self, domain, Hdomain=None):
        self.dim = np.size(domain.shape)
        assert isinstance(domain, discrs.UniformGrid)
        super().__init__(domain)
        if Hdomain is not None:
            self.Hdomain = Hdomain
        """Overload Hdomain if needed"""
        assert self.Hdomain.discr == self.domain

    def _eval(self, x):
        if self.dim==1:
            return np.sum(np.abs(gradientuniformgrid(x, spacing=self.domain.spacing)))
        else:
            return np.sum(np.linalg.norm(gradientuniformgrid(x, spacing=self.domain.spacing), axis=0))

    def _gradient(self, x):
        if self.dim==1:
            return np.sign(gradientuniformgrid(x, spacing=self.domain.spacing))
        else:
            grad = gradientuniformgrid(x, spacing=self.domain.spacing)
            grad_norm = np.linalg.norm(grad, axis=0)
            toret = np.zeros(x.shape)
            toret = np.where(grad_norm != 0, np.sum(grad, axis=0) / grad_norm, toret)
            return toret

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau, stepsize=0.1, maxiter=10):
        shape = [self.dim]+list(x.shape)
        p = np.zeros(shape)
        for i in range(maxiter):
            update = stepsize*gradientuniformgrid( self.Hdomain.gram_inv( divergenceuniformgrid(p, self.dim, spacing=self.domain.spacing))-x/tau, spacing=self.domain.spacing)
            p = (p+update) / (1+np.abs(update))
        return x-tau*divergenceuniformgrid(p, self.dim, spacing=self.domain.spacing)


"""Auxiliary method to register abstract functionals for various discretizations. Using the decorator
method described in `AbstractFunctional` does not work due to circular depenencies when
loading modules.

This is called from the `regpy` top-level module once, and can be ignored otherwise.
"""
def _register_functionals():
    HilbertNorm.register(hilbert.HilbertSpace, HilbertNormGeneric)

    L0.register(discrs.Discretization, L0Generic)

    L1.register(discrs.Discretization, L1Generic)

    TV.register(discrs.Discretization, TVGeneric)
    TV.register(discrs.UniformGrid, TVUniformGrid)
