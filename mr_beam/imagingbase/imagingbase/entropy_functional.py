import numpy as np
from regpy.functionals import Functional

class StixFunctional(Functional):
    def __init__(self, handler, domain):
        assert handler.regterm 
        assert handler.d == 'simple'
        self.handler = handler
        super().__init__(domain)

    def _eval(self, imvec):
        return self.handler._reg(imvec)

    def _gradient(self, imvec):
        return self.handler._reggrad(imvec)

    def _proximal(self, imvec, tau, f=1, m=1, niter=250, tol=10**(-10)):
        # INITIALIZATION OF THE DYKSTRA - LIKE SPLITTING
        x = imvec[:]
        p = np.zeros_like(x)
        q = np.zeros_like(x)

        for i in range(niter):

            tmp = x + p
            # Projection on the hyperplane that represents the flux constraint
            y = tmp + (f - tmp.sum()) / tmp.size
            p = x + p - y

            x = proximal_entropy(y + q, m, tau)

            if np.abs(x.sum() - f) <= 0.01 * f:
                break

            q = y + q - x

        return x

def proximal_entropy(y, m, tau, tol=10**-10, unit=1):
    # INITIALIZATION OF THE BISECTION METHOD
    # TODO where does this number come from
    a = np.full_like(y, 1e-24*unit)
    b = np.where(y > m, y, m)

    while np.max(b - a) > tol*unit:
        c = (a + b) / 2
        f_c = c - y + tau * np.log(c / m)

        tmp1 = f_c <= 0
        tmp2 = f_c >= 0

        a[tmp1] = c[tmp1]
        b[tmp2] = c[tmp2]

    c = (a + b) / 2
    return c


















