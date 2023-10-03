import numpy as np

from regpy.discrs import UniformGrid
from regpy.discrs.obstacles import StarTrigDiscr
from regpy.operators import Operator


class Potential(Operator):
    r"""Operator that maps the shape of a homogeneous heat source to the heat flux measured at some
    circle outside of the object. The heat distributions satisfies

    \[
        \begin{cases}
            \Delta u = 1_K & \text{in } \Omega \\
            u = 0          & \text{on } \partial\Omega
        \end{cases}
    \]

    where \(\partial\Omega\) is the measurement circle and \(K\) is the heat source. The operator
    maps the shape of the heat source to the Neumann data:

    \[
        \partial K \mapsto \left.\frac{\partial u}{\partial\nu}\right|_{\partial\Omega}.
    \]

    Attributes
    ----------
    domain : StarTrigDiscr
        The domain that represents the boundary curves. Actually, any star shaped curve
        discretization that can compute derivatives along the curve and derivatives wrt. coefficient
        perturbations works, but `StarTrigDiscr` is the only implementation currently available.
    radius : float
        The radius of the measurement circle.
    nmeas : int
        The number of equispaced measurement points on the circle.
    nforward : int, optional
        The order of the Fourier expansion in the forward solver.

    Raises
    ------
    ValueError
        Will be raised on evaluating the operator if the object radius is negative or penetrates
        the measurement circle.

    References
    ----------
    - F. Hettlich & W. Rundell "Iterative methods for the reconstruction of an inverse potential
      problem", Inverse Problems 12 (1996) 251–266
    - sec. 3 in T. Hohage "Logarithmic convergence rates of the iteratively regularized
      Gauss–Newton method for an inverse potential and an inverse scattering problem" Inverse
      Problems 13 (1997) 1279–1299
    """

    def __init__(self, domain, radius, nmeas, nforward=64):
        assert isinstance(domain, StarTrigDiscr)
        self.radius = radius
        """The measurement radius."""
        self.nforward = nforward
        """The Fourier order of the forward solver."""

        super().__init__(
            domain=domain,
            codomain=UniformGrid(np.linspace(0, 2 * np.pi, nmeas, endpoint=False))
        )

        k = 1 + np.arange(self.nforward)
        k_t = np.outer(k, np.linspace(0, 2 * np.pi, self.nforward, endpoint=False))
        k_tfl = np.outer(k, self.codomain.coords[0])
        self.cosin = np.cos(k_t)
        self.sinus = np.sin(k_t)
        self.cos_fl = np.cos(k_tfl)
        self.sin_fl = np.sin(k_tfl)

    def _eval(self, x, differentiate=False):
        nfwd = self.nforward
        # TODO This operator does not actually need to know about the curve at all,
        #  it is effectively a composition with eval_curve (which is not a proper Operator yet). If
        #  similar patterns work for the other obstacle operators, this should be made manifest.
        self._bd = self.domain.eval_curve(x, nvals=nfwd)
        q = self._bd.radius[0]
        if q.max() >= self.radius:
            raise ValueError('Object penetrates measurement circle')
        if q.min() <= 0:
            raise ValueError('Radial function negative')

        qq = q**2
        flux = 1 / (2 * self.radius * nfwd) * np.sum(qq) * self.codomain.ones()
        fac = 2 / (nfwd * self.radius)
        for j in range(0, (nfwd - 1) // 2):
            fac /= self.radius
            qq *= q
            flux += (
                (fac / (j + 3)) * self.cos_fl[j, :] * np.sum(qq * self.cosin[j, :]) +
                (fac / (j + 3)) * self.sin_fl[j, :] * np.sum(qq * self.sinus[j, :])
            )

        if nfwd % 2 == 0:
            fac /= self.radius
            qq *= q
            flux += fac * self.cos_fl[:, nfwd // 2] * np.sum(qq * self.cosin[nfwd // 2, :])
        return flux

    def _derivative(self, h):
        nfwd = self.nforward
        q = self._bd.radius[0]
        qqh = q * self._bd.derivative(h)

        der = 1 / (self.radius * nfwd) * np.sum(qqh) * self.codomain.ones()
        fac = 2 / (nfwd * self.radius)
        for j in range((nfwd - 1) // 2):
            fac /= self.radius
            qqh *= q
            der += fac * (
                self.cos_fl[j, :] * np.sum(qqh * self.cosin[j, :]) +
                self.sin_fl[j, :] * np.sum(qqh * self.sinus[j, :])
            )

        if nfwd % 2 == 0:
            fac /= self.radius
            qqh *= q
            der += fac * self.cos_fl[nfwd // 2, :] * np.sum(qqh * self.cosin[nfwd // 2, :])
        return der

    def _adjoint(self, g):
        nfwd = self.nforward
        q = self._bd.radius[0]
        qq = q.copy()

        adj = 1 / (self.radius * nfwd) * np.sum(g) * qq
        fac = 2 / (nfwd * self.radius)
        for j in range((nfwd - 1) // 2):
            fac /= self.radius
            qq *= q
            adj += fac * (
                np.sum(g * self.cos_fl[j, :]) * (self.cosin[j, :] * qq) +
                np.sum(g * self.sin_fl[j, :]) * (self.sinus[j, :] * qq)
            )

        if nfwd % 2 == 0:
            fac /= self.radius
            qq *= q
            adj += fac * np.sum(g * self.cos_fl[nfwd // 2, :]) * (self.cosin[nfwd // 2, :] * qq)

        return self._bd.adjoint(adj)
