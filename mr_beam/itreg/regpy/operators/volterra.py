import numpy as np

from regpy.operators import Operator
from regpy.discrs import UniformGrid


class Volterra(Operator):
    r"""The discrete Volterra operator. The Volterra operator \(V_n\) is defined as

    \[ (V_n f)(x) = \int_0^x f(t)^n dt. \]

    Its discrete form, using a Riemann sum, is simply

    \[ (V_n x)_i = h \sum_{j \leq i} x_j^n, \]

    where \(h\) is the grid spacing. \(V_1\) is linear.

    Parameters
    ----------
    domain : regpy.discrs.UniformGrid
        The domain on which the operator is defined. Must be one-dimensional.
    exponent : float
        The exponent \(n\). Default is 1.
    """

    def __init__(self, domain, exponent=1):
        assert isinstance(domain, UniformGrid)
        assert domain.ndim == 1
        self.exponent = exponent
        """The exponent."""
        super().__init__(domain, domain, linear=(exponent == 1))

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = self.exponent * x**(self.exponent - 1)
        return self.domain.volume_elem * np.cumsum(x**self.exponent)

    def _derivative(self, x):
        return self.domain.volume_elem * np.cumsum(self._factor * x)

    def _adjoint(self, y):
        x = self.domain.volume_elem * np.flipud(np.cumsum(np.flipud(y)))
        if self.linear:
            return x
        else:
            return self._factor * x
