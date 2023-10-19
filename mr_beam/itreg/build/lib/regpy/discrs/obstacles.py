import numpy as np

from regpy.discrs import UniformGrid
import regpy.util as util


# The coefficients of this curve are actually equidistant samples of the radial function,
# so we can simply inherit from UniformGrid to get the Sobolev implementation for free.
class StarTrigDiscr(UniformGrid):
    """A discretization representing star-shaped obstacles parametrized in a trigonometric basis.
    Will always be 1d an complex.

    Parameters
    ----------
    n : int
        The number of coefficients.
    """
    def __init__(self, n):
        assert isinstance(n, int)
        super().__init__(np.linspace(0, 2 * np.pi, n, endpoint=False))

    def eval_curve(self, coeffs, nvals=None, nderivs=0):
        """Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `StarTrigCurve`, which see.
        """
        return StarTrigCurve(self, coeffs, nvals, nderivs)

    def sample(self, f):
        return np.asarray(
            np.broadcast_to(f(np.linspace(0, 2*np.pi, self.size, endpoint=False)), self.shape),
            dtype=self.dtype
        )


class StarTrigCurve:
    # TODO Rename attributes. `q`, `z`, `zpabs`, etc are not good names.

    """A class representing star shaped 2d curves with radial function parametrized in a
    trigonometric basis. Should usually be instantiated via `StarTrigDiscr.eval_curve`.

    Parameters
    ----------
    discr : StarTrigDiscr
        The underlying discretization.
    coeffs : array-like
        The coefficient array of the radial function.
    nvals : int, optional
        How many points on the curve to compute. The points will be at equispaced angles in
        `[0, 2pi)`. If omitted, the number of points will match the number of `coeffs`.
    nderivs : int, optional
        How many derivatives to compute. At most 3 derivatives are implemented.
    """

    def __init__(self, discr, coeffs, nvals=None, nderivs=0):
        assert isinstance(nderivs, int) and 0 <= nderivs <= 3
        self.discr = discr
        """The discretization."""
        self.coeffs = coeffs
        """The coefficients."""
        self.nvals = nvals or self.discr.size
        """The number of computed values."""
        self.nderivs = nderivs
        """The number of computed derivatives."""

        self._frqs = 1j * np.arange(self.discr.size // 2 + 1)
        # (nvals / nx) * irfft(rfft(x), nvals) can be used for trig interpolation
        self.radius = (self.nvals / self.discr.size) * np.fft.irfft(
            (self._frqs ** np.arange(self.nderivs + 1)[:, np.newaxis]) * np.fft.rfft(coeffs),
            self.nvals,
            axis=1
        )
        """The values of the radial function and its derivatives, shaped `(nderivs + 1, nvals)`."""

        t = np.linspace(0, 2 * np.pi, self.nvals, endpoint=False)
        cost = np.cos(t)
        sint = np.sin(t)

        self.curve = np.zeros((self.nderivs + 1, 2, self.nvals))
        """The points on the curve and its derivatives, shaped `(nderivs + 1, 2, nvals)`."""
        binom = np.ones(self.nderivs + 1, dtype=int)
        for n in range(self.nderivs + 1):
            binom[1:n] += binom[:n-1]
            aux = binom[:n+1, np.newaxis] * self.radius[n::-1]
            even = np.sum(aux[::4], axis=0) - np.sum(aux[2::4], axis=0)
            odd = np.sum(aux[1::4], axis=0) - np.sum(aux[3::4], axis=0)
            self.curve[n, 0] = even * cost - odd * sint
            self.curve[n, 1] = even * sint + odd * cost

        if self.nderivs == 0:
            return

        self.normal = np.stack([self.curve[1, 1], -self.curve[1, 1]])
        """The (unnormalized) outer normal vector as `(2, nvals)` array. Its norm identical to that
        of the tangent vector `curve[1]`."""
        self.tangent_norm = np.linalg.norm(self.normal, axis=0)
        """The absolute values of the tangent and normal vectors as `(nvals,)` array."""

    # TODO Should these be turned into operators?

    def derivative(self, h):
        return (self.nvals / self.discr.size) * np.fft.irfft(
            np.fft.rfft(h), self.nvals
        )

    def adjoint(self, g):
        return (self.nvals / self.discr.size) * util.adjoint_rfft(
            util.adjoint_irfft(g, self.discr.size // 2 + 1),
            self.discr.size
        )

    def der_normal(self, h):
        """Computes the normal part of the perturbation of the curve caused by
        perturbing the coefficient vector curve.coeff in direction `h`."""
        return (self.radius[0] / self.tangent_norm) * self.derivative(h)

    def adjoint_der_normal(self, g):
        """Computes the adjoint of `der_normal`."""
        return self.adjoint((self.radius[0] / self.tangent_norm) * g)

    def arc_length_der(self, h):
        """Computes the derivative of `h` with respect to arclength."""
        return (self.nvals / self.discr.size) * np.fft.irfft(
            self._frqs * np.fft.rfft(h), self.nvals
        ) / self.tangent_norm
