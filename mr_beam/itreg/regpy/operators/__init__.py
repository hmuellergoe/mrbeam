"""Forward operators

This module provides the basis for defining forward operators, and implements some simple
auxiliary operators. Actual forward problems are implemented in submodules.

The base class is `Operator`.
"""

# TODO Document all instance variables, so they appear in pdoc's output.

from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import fftconvolve

from regpy import functionals, util, discrs


class _Revocable:
    def __init__(self, val):
        self.__val = val

    @classmethod
    def take(cls, other):
        return cls(other.revoke())

    def get(self):
        try:
            return self.__val
        except AttributeError:
            raise RuntimeError('Attempted to use revoked reference') from None

    def revoke(self):
        val = self.get()
        del self.__val
        return val

    @property
    def valid(self):
        try:
            self.__val
            return True
        except AttributeError:
            return False


class Operator:
    """Base class for forward operators. Both linear and non-linear operators are handled. Operator
    instances are callable, calling them with an array argument evaluates the operator.

    Subclasses implementing non-linear operators should implement the following methods:

        _eval(self, x, differentiate=False)
        _derivative(self, x)
        _adjoint(self, y)

    These methods are not intended for external use, but should be invoked indirectly via calling
    the operator or using the `Operator.linearize` method. They must not modify their argument, and
    should return arrays that can be freely modified by the caller, i.e. should not share data
    with anything. Usually, this means they should allocate a new array for the return value.

    Implementations can assume the arguments to be part of the specified discretizations, and return
    values will be checked for consistency.

    The mechanism for derivatives and their adjoints is this: whenever a derivative is to be
    computed, `_eval` will be called first with `differentiate=True`, and should produce the
    operator's value and perform any precomputation needed for evaluating the derivative. Any
    subsequent invocation of `_derivative` and `_adjoint` should evaluate the derivative or its
    adjoint at the same point `_eval` was called. The reasoning is this

    - In most cases, the derivative alone is not useful. Rather, one needs a linearization of the
      operator around some point, so the value is almost always needed.
    - Many expensive computations, e.g. assembling of finite element matrices, need to be carried
      out only once per linearization point, and can be shared between the operator and the
      derivative, so they should only be computed once (in `_eval`).

    For callers, this means that since the derivative shares data with the operator, it can't be
    reliably called after the operator has been evaluated somewhere else, since shared data may
    have been overwritten. The `Operator`, `Derivative` and `Adjoint` classes ensure that an
    exception is raised when an invalidated derivative is called.

    If derivatives at multiple points are needed, a copy of the operator should be performed using
    `copy.deepcopy`. For efficiency, subclasses can add the names of attributes that are considered
    as constants and should not be deepcopied to `self._consts` (a `set`). By default, `domain` and
    `codomain` will not be copied, since `regpy.discrs.Discretization` instances should never
    change in-place.

    If no derivative at some point is needed, `_eval` will be called with `differentiate=False`,
    allowing it to save on precomputations. It does not need to ensure that data shared with some
    derivative remains intact; all derivative instances will be invalidated regardless.

    Linear operators should implement

        _eval(self, x)
        _adjoint(self, y)

    Here the logic is simpler, and no sharing of precomputations is needed (unless it applies to the
    operator as a whole, in which case it should be performed in `__init__`).

    Note that the adjoint should be computed with respect to the standard real inner product on the
    domain / codomain, given as

        np.real(np.vdot(x, y))

    Other inner product on discretizations are independent of both discretizations and operators,
    and are implemented in the `regpy.hilbert` module.

    Basic operator algebra is supported:

        a * op1 + b * op2    # linear combination
        op1 * op2            # composition
        arr * op             # composition with array multiplication in codomain
        op * arr             # composition with array multiplication in domain
        op + arr             # operator shifted in codomain
        op + scalar          # dto.

    Parameters
    ----------
    domain, codomain : regpy.discrs.Discretization or None
        The discretization on which the operator's arguements / values are defined. Using `None`
        suppresses some consistency checks and is intended for ease of development, but should be
        not be used except as a temporary measure. Some constructions like direct sums will fail
        if the discretizations are unknown.
    linear : bool, optional
        Whether the operator is linear. Default: `False`.
    """

    log = util.classlogger

    def __init__(self, domain=None, codomain=None, linear=False):
        assert not domain or isinstance(domain, discrs.Discretization)
        assert not codomain or isinstance(codomain, discrs.Discretization)
        self.domain = domain
        """The discretization on which the operator is defined. Either a
        subclass of `regpy.discrs.Discretization` or `None`."""
        self.codomain = codomain
        """The discretization on which the operator values are defined. Either
        a subclass of `regpy.discrs.Discretization` or `None`."""
        self.linear = linear
        """Boolean indicating whether the operator is linear."""
        self._consts = {'domain', 'codomain'}

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self._consts:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def attrs(self):
        """The set of all instance attributes. Useful for updating the `_consts` attribute via

            self._consts.update(self.attrs)

        to declare every current attribute as constant for deep copies.
        """
        return set(self.__dict__)

    def __call__(self, x):
        assert not self.domain or x in self.domain
        if self.linear:
            y = self._eval(x)
        else:
            self.__revoke()
            y = self._eval(x, differentiate=False)
        assert not self.codomain or y in self.codomain
        return y

    def linearize(self, x):
        """Linearize the operator around some point.

        Parameters
        ----------
        x : array-like
            The point around which to linearize.

        Returns
        -------
        array, Derivative
            The value and the derivative at `x`, the latter as an `Operator` instance.
        """
        if self.linear:
            return self(x), self
        else:
            assert not self.domain or x in self.domain
            self.__revoke()
            y = self._eval(x, differentiate=True)
            assert not self.codomain or y in self.codomain
            deriv = Derivative(self.__get_handle())
            return y, deriv

    @util.memoized_property
    def adjoint(self):
        """For linear operators, this is the adjoint as a linear `regpy.operators.Operator`
        instance. Will only be computed on demand and saved for subsequent invocations.
        """
        return Adjoint(self)

    def __revoke(self):
        try:
            self.__handle = _Revocable.take(self.__handle)
        except AttributeError:
            pass

    def __get_handle(self):
        try:
            return self.__handle
        except AttributeError:
            self.__handle = _Revocable(self)
            return self.__handle

    def _eval(self, x, differentiate=False):
        raise NotImplementedError

    def _derivative(self, x):
        raise NotImplementedError

    def _adjoint(self, y):
        raise NotImplementedError

    @property
    def inverse(self):
        """A property containing the  inverse as an `Operator` instance. In most cases this will
        just raise a `NotImplementedError`, but subclasses may override this if possible and useful.
        To avoid recomputing the inverse on every access, `regpy.util.memoized_property` may be
        useful."""
        raise NotImplementedError

    def norm(self, iterations=10):
        """For linear operators, estimate the operator norm with respect to the standard norm on its
        domain / codomain using the power method.

        Parameters
        ----------
        iterations : int, optional
            The number of iterations. Default: 10.

        Returns
        -------
        float
            An estimate for the operator norm.
        """
        assert self.linear
        h = self.domain.rand()
        norm = np.sqrt(np.real(np.vdot(h, h)))
        for _ in range(iterations):
            h = h / norm
            # TODO gram matrices
            h = self.adjoint(self(h))
            norm = np.sqrt(np.real(np.vdot(h, h)))
        return np.sqrt(norm)

    def __mul__(self, other):
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, Operator):
            return Composition(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return self * Multiplication(self.domain, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator):
            return Composition(other, self)
        elif np.isscalar(other):
            if other == 1:
                return self
            else:
                return LinearCombination((other, self))
        elif isinstance(other, np.ndarray):
            return Multiplication(self.codomain, other) * self
        else:
            return NotImplemented

    def __add__(self, other):
        if np.isscalar(other) and other == 0:
            return self
        elif isinstance(other, Operator):
            return LinearCombination(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return Shifted(self, other)
        else:
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


class Adjoint(Operator):
    """An proxy class wrapping a linear operator. Calling it will evaluate the operator's
    adjoint. This class should not be instantiated directly, but rather through the
    `Operator.adjoint` property of a linear operator.
    """

    def __init__(self, op):
        assert op.linear
        self.op = op
        """The underlying operator."""
        super().__init__(op.codomain, op.domain, linear=True)

    def _eval(self, x):
        return self.op._adjoint(x)

    def _adjoint(self, x):
        return self.op._eval(x)

    @property
    def adjoint(self):
        return self.op

    @property
    def inverse(self):
        return self.op.inverse.adjoint

    def __repr__(self):
        return util.make_repr(self, self.op)


class Derivative(Operator):
    """An proxy class wrapping a non-linear operator. Calling it will evaluate the operator's
    derivative. This class should not be instantiated directly, but rather through the
    `Operator.linearize` method of a non-linear operator.
    """

    def __init__(self, op):
        if not isinstance(op, _Revocable):
            # Wrap plain operators in a _Revocable that will never be revoked to
            # avoid case distinctions below.
            op = _Revocable(op)
        self.op = op
        """The underlying operator."""
        _op = op.get()
        super().__init__(_op.domain, _op.codomain, linear=True)

    def _eval(self, x):
        return self.op.get()._derivative(x)

    def _adjoint(self, x):
        return self.op.get()._adjoint(x)

    def __repr__(self):
        return util.make_repr(self, self.op.get())


class LinearCombination(Operator):
    """A linear combination of operators. This class should normally not be instantiated directly,
    but rather through adding and multipliying `Operator` instances and scalars.
    """

    def __init__(self, *args):
        coeff_for_op = defaultdict(lambda: 0)
        for arg in args:
            if isinstance(arg, tuple):
                coeff, op = arg
            else:
                coeff, op = 1, arg
            assert isinstance(op, Operator)
            assert (
                not np.iscomplex(coeff)
                or not op.codomain
                or op.codomain.is_complex
            )
            if isinstance(op, type(self)):
                for c, o in zip(op.coeffs, op.ops):
                    coeff_for_op[o] += coeff * c
            else:
                coeff_for_op[op] += coeff
        self.coeffs = []
        """List of coefficients of the combined operators."""
        self.ops = []
        """List of combined operators."""
        for op, coeff in coeff_for_op.items():
            self.coeffs.append(coeff)
            self.ops.append(op)

        domains = [op.domain for op in self.ops if op.domain]
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains)
        else:
            domain = None

        codomains = [op.codomain for op in self.ops if op.codomain]
        if codomains:
            codomain = codomains[0]
            assert all(c == codomain for c in codomains)
        else:
            codomain = None

        super().__init__(domain, codomain, linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False):
        y = self.codomain.zeros()
        if differentiate:
            self._derivs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if differentiate:
                z, deriv = op.linearize(x)
                self._derivs.append(deriv)
            else:
                z = op(x)
            y += coeff * z
        return y

    def _derivative(self, x):
        y = self.codomain.zeros()
        for coeff, deriv in zip(self.coeffs, self._derivs):
            y += coeff * deriv(x)
        return y

    def _adjoint(self, y):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        x = self.domain.zeros()
        for coeff, op in zip(self.coeffs, ops):
            x += np.conj(coeff) * op.adjoint(y)
        return x

    @property
    def inverse(self):
        if len(self.ops) > 1:
            raise NotImplementedError
        return (1 / self.coeffs[0]) * self.ops[0].inverse

    def __repr__(self):
        return util.make_repr(self, *zip(self.coeffs, self.ops))

    def __str__(self):
        reprs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if coeff == 1:
                reprs.append(repr(op))
            else:
                reprs.append('{} * {}'.format(coeff, op))
        return ' + '.join(reprs)


class Composition(Operator):
    """A composition of operators. This class should normally not be instantiated directly,
    but rather through multipliying `Operator` instances.
    """

    def __init__(self, *ops):
        for f, g in zip(ops, ops[1:]):
            assert not f.domain or not g.codomain or f.domain == g.codomain
        self.ops = []
        """The list of composed operators."""
        for op in ops:
            assert isinstance(op, Operator)
            if isinstance(op, Composition):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        super().__init__(
            self.ops[-1].domain, self.ops[0].codomain,
            linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False):
        y = x
        if differentiate:
            self._derivs = []
            for op in self.ops[::-1]:
                y, deriv = op.linearize(y)
                self._derivs.insert(0, deriv)
        else:
            for op in self.ops[::-1]:
                y = op(y)
        return y

    def _derivative(self, x):
        y = x
        for deriv in self._derivs[::-1]:
            y = deriv(y)
        return y

    def _adjoint(self, y):
        x = y
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        for op in ops:
            x = op.adjoint(x)
        return x

    @util.memoized_property
    def inverse(self):
        return Composition(*(op.inverse for op in self.ops[::-1]))

    def __repr__(self):
        return util.make_repr(self, *self.ops)


class Identity(Operator):
    """The identity operator on a discretization. Performs a copy to prevent callers from
    accidentally modifying the argument when modifying the return value.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization.
    """

    def __init__(self, domain):
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return x.copy()

    def _adjoint(self, x):
        return x.copy()

    @property
    def inverse(self):
        return self

    def __repr__(self):
        return util.make_repr(self, self.domain)


class CholeskyInverse(Operator):
    """Implements the inverse of a linear, self-adjoint operator via Cholesky decomposition. Since
    it needs to assemble a full matrix, this should not be used for high-dimensional operators.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator to invert.
    matrix : array-like, optional
        If a matrix of `op` is already available, it can be passed in to avoid recomputation.
    """
    def __init__(self, op, matrix=None):
        assert op.linear
        assert op.domain and op.domain == op.codomain
        domain = op.domain
        if matrix is None:
            matrix = np.empty((domain.realsize,) * 2, dtype=float)
            for j, elm in enumerate(domain.iter_basis()):
                matrix[j, :] = domain.flatten(op(elm))
        self.factorization = cho_factor(matrix)
        """The Cholesky factorization for use with `scipy.linalg.cho_solve`"""
        super().__init__(
            domain=domain,
            codomain=domain,
            linear=True
        )
        self.op = op

    def _eval(self, x):
        return self.domain.fromflat(
            cho_solve(self.factorization, self.domain.flatten(x)))

    def _adjoint(self, x):
        return self._eval(x)

    @property
    def inverse(self):
        """Returns the original operator."""
        return self.op

    def __repr__(self):
        return util.make_repr(self, self.op)


class CoordinateProjection(Operator):
    """A projection operator onto a subset of the domain. The codomain is a one-dimensional
    `regpy.discrs.Discretization` of the same dtype as the domain.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization
    mask : array-like
        Boolean mask of the subset onto which to project.
    """
    def __init__(self, domain, mask):
        mask = np.broadcast_to(mask, domain.shape)
        assert mask.dtype == bool
        self.mask = mask
        super().__init__(
            domain=domain,
            codomain=discrs.Discretization(np.sum(mask), dtype=domain.dtype),
            linear=True
        )

    def _eval(self, x):
        return x[self.mask]

    def _adjoint(self, x):
        y = self.domain.zeros()
        y[self.mask] = x
        return y

    def __repr__(self):
        return util.make_repr(self, self.domain, self.mask)

class CoordinateMask(Operator):
    """A projection operator onto a subset of the domain. The remaining array elements are set to zero.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization
    mask : array-like
        Boolean mask of the subset onto which to project.
    """
    def __init__(self, domain, mask):
        self.mask = mask
        super().__init__(
            domain=domain,
            codomain=domain,
            linear=True
        )

    def _eval(self, x):
        return np.where(self.mask==False, 0, x)

    def _adjoint(self, x):
        return np.where(self.mask==False, 0, x)

    def __repr__(self):
        return util.make_repr(self, self.domain, self.mask)


class Multiplication(Operator):
    """A multiplication operator by a constant factor.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization
    factor : array-like
        The factor by which to multiply. Can be anything that can be broadcast to `domain.shape`.
    """
    def __init__(self, domain, factor):
        factor = np.asarray(factor)
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        if domain:
            factor = np.broadcast_to(factor, domain.shape)
            assert factor in domain
        self.factor = factor
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.factor * x

    def _adjoint(self, x):
        if self.domain.is_complex:
            return np.conj(self.factor) * x
        else:
            # Avoid conj() when not needed (performs copy)
            # TODO should we just store conj(factor) once?
            return self.factor * x

    @util.memoized_property
    def inverse(self):
        sav = np.seterr(divide='raise')
        try:
            return Multiplication(self.domain, 1 / self.factor)
        finally:
            np.seterr(**sav)

    def __repr__(self):
        return util.make_repr(self, self.domain, self.factor)


class Shifted(Operator):
    """Shift an operator by a constant offset in the codomain.

    Parameters
    ----------
    op : Operator
        The underlying operator.
    offset : array-like
        The offset by which to shift. Can be anything that can be broadcast to `op.codomain.shape`.
    """
    def __init__(self, op, offset):
        assert offset in op.codomain
        super().__init__(op.domain, op.codomain)
        if isinstance(op, type(self)):
            offset = offset + op.offset
            op = op.op
        self.op = op
        self.offset = offset

    def _eval(self, x, differentiate=False):
        if differentiate:
            y, self._deriv = self.op.linearize(x)
            return y + self.offset
        else:
            return self.op(x) + self.offset

    def _derivative(self, x):
        return self._deriv(x)

    def _adjoint(self, y):
        return self._deriv.adjoint(y)


class FourierTransform(Operator):
    def __init__(self, domain, centered=False, axes=None):
        assert isinstance(domain, discrs.UniformGrid)
        frqs = domain.frequencies(centered=centered, axes=axes)
        if centered:
            codomain = discrs.UniformGrid(*frqs, dtype=complex)
        else:
            # In non-centered case, the frequencies are not ascencing, so even using Grid here is slighty questionable.
            codomain = discrs.Grid(*frqs, dtype=complex)
        super().__init__(domain, codomain, linear=True)
        self.centered = centered
        self.axes = axes

    def _eval(self, x):
        if self.centered:
            x = np.fft.ifftshift(x, axes=self.axes)
        y = np.fft.fftn(x, axes=self.axes, norm='ortho')
        if self.centered:
            return np.fft.fftshift(y, axes=self.axes)
        else:
            return y

    def _adjoint(self, y):
        if self.centered:
            y = np.fft.ifftshift(y, axes=self.axes)
        x = np.fft.ifftn(y, axes=self.axes, norm='ortho')
        if self.centered:
            x = np.fft.fftshift(x, axes=self.axes)
        if self.domain.is_complex:
            return x
        else:
            return np.real(x)

    @property
    def inverse(self):
        return self.adjoint

    def __repr__(self):
        return util.make_repr(self, self.domain)
    
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

class MatrixMultiplication(Operator):
    """Implements a matrix multiplication with a given matrix. Domain and codomain are plain
    `regpy.discrs.Discretization` instances.

    Parameters
    ----------
    matrix : array-like
        The matrix.
    inverse : Operator, array-like, 'inv', 'cholesky' or None
        How to implement the inverse operator. If available, this should be given as `Operator`
        or array. If `'inv'`, `numpy.linalg.inv` will be used. If `'cholesky'`, a
        `CholeskyInverse` instance will be returned.
    """

    # TODO complex case
    def __init__(self, matrix, inverse=None, domain=None, codomain=None):
        self.matrix = matrix
        super().__init__(
            domain=domain or discrs.Discretization(matrix.shape[1]),
            codomain=codomain or discrs.Discretization(matrix.shape[0]),
            linear=True
        )
        self._inverse = inverse

    def _eval(self, x):
        return self.matrix @ x

    def _adjoint(self, y):
        return self.matrix.T @ y

    @util.memoized_property
    def inverse(self):
        if isinstance(self._inverse, Operator):
            return self._inverse
        elif isinstance(self._inverse, np.ndarray):
            return MatrixMultiplication(self._inverse, inverse=self)
        elif isinstance(self._inverse, str):
            if self._inverse == 'inv':
                return MatrixMultiplication(np.linalg.inv(self.matrix), inverse=self)
            if self._inverse == 'cholesky':
                # TODO LU, QR
                return CholeskyInverse(self, matrix=self.matrix)
        raise NotImplementedError

    def __repr__(self):
        return util.make_repr(self, self.matrix)


class Power(Operator):
    r"""The operator \(x \mapsto x^n\).

    Parameters
    ----------
    power : float
        The exponent.
    domain : regpy.discrs.Discretization
        The underlying discretization
    """

    # TODO complex case
    def __init__(self, power, domain):
        self.power = power
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = self.power * x**(self.power - 1)
        return x**self.power

    def _derivative(self, x):
        return self._factor * x

    def _adjoint(self, y):
        return self._factor * y


class DirectSum(Operator):
    """The direct sum of operators. For

        T_i : X_i -> Y_i

    the direct sum

        T := DirectSum(T_i) : DirectSum(X_i) -> DirectSum(Y_i)

    is given by `T(x)_i := T_i(x_i)`. As a matrix, this is the block-diagonal
    with blocks (T_i).

    Parameters
    ----------
    *ops : tuple of Operator
    flatten : bool, optional
        If True, summands that are themselves direct sums will be merged with
        this one. Default: False.
    domain, codomain : discrs.Discretization or callable, optional
        Either the underlying discretization or a factory function that will be called with all
        summands' discretizations passed as arguments and should return a discrs.DirectSum instance.
        The resulting discretization should be iterable, yielding the individual summands.
        Default: discrs.DirectSum.
    """

    def __init__(self, *ops, flatten=False, domain=None, codomain=None):
        assert all(isinstance(op, Operator) for op in ops)
        self.ops = []
        for op in ops:
            if flatten and isinstance(op, type(self)):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)

        if domain is None:
            domain = discrs.DirectSum
        if isinstance(domain, discrs.Discretization):
            pass
        elif callable(domain):
            domain = domain(*(op.domain for op in self.ops))
        else:
            raise TypeError('domain={} is neither a Discretization nor callable'.format(domain))
        assert all(op.domain == d for op, d in zip(ops, domain))

        if codomain is None:
            codomain = discrs.DirectSum
        if isinstance(codomain, discrs.Discretization):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
        else:
            raise TypeError('codomain={} is neither a Discretization nor callable'.format(codomain))
        assert all(op.codomain == c for op, c in zip(ops, codomain))

        super().__init__(domain=domain, codomain=codomain, linear=all(op.linear for op in ops))

    def _eval(self, x, differentiate=False):
        elms = self.domain.split(x)
        if differentiate:
            linearizations = [op.linearize(elm) for op, elm in zip(self.ops, elms)]
            self._derivs = [l[1] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        else:
            return self.codomain.join(*(op(elm) for op, elm in zip(self.ops, elms)))

    def _derivative(self, x):
        elms = self.domain.split(x)
        return self.codomain.join(
            *(deriv(elm) for deriv, elm in zip(self._derivs, elms))
        )

    def _adjoint(self, y):
        elms = self.codomain.split(y)
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        return self.domain.join(
            *(op.adjoint(elm) for op, elm in zip(ops, elms))
        )

    @util.memoized_property
    def inverse(self):
        """The component-wise inverse as a `DirectSum`, if all of them exist."""
        return DirectSum(
            *(op.inverse for op in self.ops),
            domain=self.codomain,
            codomain=self.domain
        )

    def __repr__(self):
        return util.make_repr(self, *self.ops)

    def __getitem__(self, item):
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)


class Exponential(Operator):
    r"""The pointwise exponential operator.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization.
    """

    def __init__(self, domain):
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._exponential_factor = np.exp(x)
            return self._exponential_factor
        return np.exp(x)

    def _derivative(self, x):
        return self._exponential_factor * x

    def _adjoint(self, y):
        return self._exponential_factor.conj() * y
    

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
    
    
class RealPart(Operator):
    """The pointwise real part operator.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discreization. The codomain will be the corresponding
        `regpy.discrs.Discretization.real_space`.
    """

    def __init__(self, domain):
        if domain:
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return x.real.copy()

    def _adjoint(self, y):
        return y.copy()


class ImaginaryPart(Operator):
    """The pointwise imaginary part operator.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discreization. The codomain will be the corresponding
        `regpy.discrs.Discretization.real_space`.
    """

    def __init__(self, domain):
        if domain:
            assert domain.is_complex
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return x.imag.copy()

    def _adjoint(self, y):
        return 1j * y


class SquaredModulus(Operator):
    """The pointwise squared modulus operator.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discreization. The codomain will be the corresponding
        `regpy.discrs.Discretization.real_space`.
    """

    def __init__(self, domain):
        if domain:
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = 2 * x
        return x.real**2 + x.imag**2

    def _derivative(self, x):
        return (self._factor.conj() * x).real

    def _adjoint(self, y):
        return self._factor * y


class Zero(Operator):
    """The constant zero operator.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization.
    domain : regpy.discrs.Discretization, optional
        The discretization if the codomain. Defaults to `domain`.
    """
    def __init__(self, domain, codomain=None):
        if codomain is None:
            codomain = domain
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return self.codomain.zeros()

    def _adjoint(self, x):
        return self.domain.zeros()


class ApproximateHessian(Operator):
    """An approximation of the Hessian of a `regpy.functionals.Functional` at some point, computed
    using finite differences of its `regpy.functionals.Functional.gradient`.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    x : array-like
        The point at which to evaluate the Hessian.
    stepsize : float, optional
        The stepsize for the finite difference approximation.
    """
    def __init__(self, func, x, stepsize=1e-8):
        assert isinstance(func, functionals.Functional)
        self.gradx = func.gradient(x)
        """The gradient at `x`"""
        self.func = func
        self.x = x.copy()
        self.stepsize = stepsize
        # linear=True is a necessary lie
        super().__init__(func.domain, func.domain, linear=True)
        self.log.info('Using approximate Hessian of functional {}'.format(self.func))

    def _eval(self, h):
        grad = self.func.gradient(self.x + self.stepsize * h)
        return grad - self.gradx

    def _adjoint(self, x):
        return self._eval(x)
