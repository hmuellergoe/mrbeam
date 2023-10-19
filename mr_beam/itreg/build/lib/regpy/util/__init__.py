from functools import wraps
from logging import getLogger

import numpy as np
from scipy.spatial.qhull import Voronoi


@property
def classlogger(self):
    """The [`logging.Logger`][1] instance. Every subclass has a separate instance, named by its
    fully qualified name. Subclasses should use it instead of `print` for any kind of status
    information to allow users to control output formatting, verbosity and persistence.

    [1]: https://docs.python.org/3/library/logging.html#logging.Logger
    """
    return getattr(self, '_log', None) or getLogger(type(self).__qualname__)


@classlogger.setter
def classlogger(self, log):
    self._log = log


def memoized_property(prop):
    attr = '__memoized_' + prop.__qualname__

    @property
    @wraps(prop)
    def mprop(self):
        try:
            return getattr(self, attr)
        except AttributeError:
            pass
        setattr(self, attr, prop(self))
        return getattr(self, attr)

    return mprop


def set_defaults(params, **defaults):
    if params is not None:
        defaults.update(params)
    return defaults


def complex2real(z, axis=-1):
    assert is_complex_dtype(z.dtype)
    if z.flags.c_contiguous:
        x = z.view(dtype=z.real.dtype).reshape(z.shape + (2,))
    else:
        # TODO Does this actually work in all cases, or do we have to perform a
        # copy here?
        x = np.lib.stride_tricks.as_strided(
            z.real, shape=z.shape + (2,),
            strides=z.strides + (z.real.dtype.itemsize,))
    return np.moveaxis(x, -1, axis)


def real2complex(x, axis=-1):
    assert is_real_dtype(x.dtype)
    assert x.shape[axis] == 2
    x = np.moveaxis(x, axis, -1)
    if np.issubdtype(x.dtype, np.floating) and x.flags.c_contiguous:
        return x.view(dtype=np.result_type(1j, x))[..., 0]
    else:
        z = np.array(x[..., 0], dtype=np.result_type(1j, x))
        z.imag = x[..., 1]
        return z


def is_real_dtype(obj):
    if np.isscalar(obj):
        obj = np.asarray(obj)
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return (
        np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)
    )


def is_complex_dtype(obj):
    if np.isscalar(obj):
        obj = np.asarray(obj)
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return np.issubdtype(dtype, np.complexfloating)


def is_uniform(x):
    x = np.asarray(x)
    assert x.ndim == 1
    diffs = x[1:] - x[:-1]
    return np.allclose(diffs, diffs[0])


def linspace_circle(num, *, start=0, stop=None, endpoint=False):
    if not stop:
        stop = start + 2 * np.pi
    angles = np.linspace(start, stop, num, endpoint)
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def make_repr(self, *args, **kwargs):
    arglist = []
    for arg in args:
        arglist.append(repr(arg))
    for k, v in sorted(kwargs.items()):
        arglist.append("{}={}".format(repr(k), repr(v)))
    return '{}({})'.format(type(self).__qualname__, ', '.join(arglist))


eps = np.finfo(float).eps


def bounded_voronoi(nodes, left, down, up, right):
    """Computes the Voronoi diagram with a bounding box
    """

    # Extend the set of nodes by reflecting along boundaries
    nodes_left = 2 * np.array([left - 1e-6, 0]) - nodes
    nodes_down = 2 * np.array([0, down - 1e-6]) - nodes
    nodes_right = 2 * np.array([right + 1e-6, 0]) - nodes
    nodes_up = 2 * np.array([0, up + 1e-6]) - nodes

    # Compute the extended Voronoi diagram
    evor = Voronoi(np.concatenate([nodes, nodes_up, nodes_down, nodes_left, nodes_right]))

    # Shrink the Voronoi diagram
    regions = [evor.regions[reg] for reg in evor.point_region[:nodes.shape[0]]]
    used_vertices = np.unique([i for reg in regions for i in reg])
    regions = [[np.where(used_vertices == i)[0][0] for i in reg] for reg in regions]
    vertices = [evor.vertices[i] for i in used_vertices]

    return regions, vertices


def broadcast_shapes(*shapes):
    a = np.ones((max(len(s) for s in shapes), len(shapes)), dtype=int)
    for i, s in enumerate(shapes):
        a[-len(s):, i] = s
    result = np.max(a, axis=1)
    for r, x in zip(result, a):
        if np.any((x != 1) & (x != r)):
            raise ValueError('Shapes can not be broadcast')
    return result


def trig_interpolate(val, n):
    # TODO get rid of fftshift
    """Computes `n` Fourier coeffients to the point values given by by `val`
    such that `ifft(fftshift(coeffs))` is an interpolation of `val`.
    """
    if n % 2 != 0:
        ValueError('n should be even')
    N = len(val)
    coeffhat = np.fft.fft(val)
    coeffs = np.zeros(n, dtype=complex)
    if n >= N:
        coeffs[:N // 2] = coeffhat[:N // 2]
        coeffs[-(N // 2) + 1:] = coeffhat[N // 2 + 1:]
        if n > N:
            coeffs[N // 2] = 0.5 * coeffhat[N // 2]
            coeffs[-(N // 2)] = 0.5 * coeffhat[N // 2]
        else:
            coeffs[N // 2] = coeffhat[N // 2]
    else:
        coeffs[:n // 2] = coeffhat[:n // 2]
        coeffs[n // 2 + 1:] = coeffhat[-(n // 2) + 1:]
        coeffs[n // 2] = 0.5 * (coeffhat[n // 2] + coeffhat[-(n // 2)])
    coeffs = n / N * np.fft.ifftshift(coeffs)
    return coeffs


def adjoint_rfft(y, size, n=None):
    """Compute the adjoint of `numpy.fft.rfft`. More concretely, the adjoint of

        x |-> rfft(x, n)

    is

        y |-> adjoint_rfft(y, x.size, n)

    Since the size of `x` can not be determined from `y` and `n`, it needs to be given explicitly.

    Parameters
    ----------
    y : array-like
        The input array.
    size : int
        The size of the output, i.e. the size of the original input to `rfft`.
    n : int, optional
        The same `n` given as optional parameter to `rfft`. If omitted, `size` will be used, so
        `adjoint_rfft(y, x.size)` is adjoint to `rfft(x)`.

    Returns
    -------
    array of shape (size,)
    """

    if n is None:
        n = size
    # y needs to be in the image of x |-> rfft(x, n), so its size must be this:
    assert n // 2 + 1 == y.size
    # The following is the adjoint of x |-> rfft(x, n=x.size), i.e. the "natural size" rfft,
    # of size n
    result = np.fft.irfft(y, n)
    result *= n / 2
    result += y[0].real / 2
    if n % 2 == 0:
        aux = y[-1].real / 2
        result[::2] += aux
        result[1::2] -= aux
    # The general case is
    #     rfft(x, n) = rfft(p(x, n), p(x, n).size)
    # where p is a truncation or padding operator (and p(x, n).size == n), so the adjoint is the
    # "natural" adjoint followed by (the adjoint of the self-adjoint) padding or truncation.
    if n == size:
        return result
    elif size < n:
        return result[:size]
    else:
        aux = np.zeros(size, dtype=result.dtype)
        aux[:n] = result
        return aux


def adjoint_irfft(y, size=None):
    """Compute the adjoint of `numpy.fft.irfft`. More concretely, the adjoint of

        x |-> irfft(x, n)

    is

        y |-> adjoint_irfft(y, x.size)

    Since the size of `x` can not be determined from `y`, it needs to be given explicitly. The
    parameter `n`, however, is determined as the output size of `irfft`, so it does not not need to
    be specified for the adjoint.

    Parameters
    ----------
    y : array-like
        The input array.
    size : int, optional
        The size of the output, i.e. the size of the original input to `irfft`. If omitted,
        `x.size // 2 + 1` will be used, i.e. we assume the `irfft` is inverse to a plain `rfft(x)`,
        without additional padding or truncation.

    Returns
    -------
    array of shape (size,)
    """

    if size is None:
        size = y.size // 2 + 1
    # We proceed as in `adjoint_rfft`: first compute the adjoint of the "natural size" `irfft` by
    # using `rfft(x)` without explicit `n`...
    result = np.fft.rfft(y)
    result[0] -= np.sum(y) / 2
    if y.size % 2 == 0:
        result[-1] -= (np.sum(y[::2]) - np.sum(y[1::2])) / 2
    result *= 2 / y.size
    # ... then pad or truncate.
    if size == result.size:
        return result
    elif size < result.size:
        return result[:size]
    else:
        aux = np.zeros(size, dtype=result.dtype)
        aux[:result.size] = result
        return aux


def foo(n, m):
    x = np.random.randn(n) + 1j * np.random.randn(n)
    fx = np.fft.irfft(x, m)
    y = np.random.randn(fx.size)
    fty = adjoint_irfft(y, x.size)
    return np.real(np.vdot(y, fx)) - np.real(np.vdot(fty, x))


def asdf():
    a = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            a[i, j] = foo(i + 6, j + 6)
    return a
    
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
    