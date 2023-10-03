import numpy as np
from scipy.sparse import linalg as spla

from regpy.solvers import Solver


class NewtonCG(Solver):
    """The Newton-CG method. Solves the potentially non-linear, ill-posed equation:

        T(x) = y,

    where T is a Frechet-differentiable operator. The Newton equations are solved by the
    conjugate gradient method applied to the normal equation (CGNE) using the regularizing
    properties of CGNE with early stopping (see Hanke 1997).
    """

    def __init__(self, op, data, init, cgmaxit=50, rho=0.8):
        super().__init__()
        self.op = op
        self.data = data
        self.x = init
        self._outer_update()
        self.rho = rho
        self.cgmaxit = cgmaxit

    def _outer_update(self):
        self._x_k = np.zeros(np.shape(self.x))
        self.y = self.op(self.x)
        self._residual = self.data - self.y
        _, self.deriv = self.op.linearize(self.x)
        self._s = self._residual - self.deriv(self._x_k)
        self._s2 = self.op.codomain.gram(self._s)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.op.domain.gram_inv(self._rtilde)
        self._d = self._r
        self._innerProd = self.op.domain.inner(self._r, self._rtilde)
        self._norms0 = np.sqrt(np.real(self.op.domain.inner(self._s2, self._s)))
        self._k = 1

    def _inner_update(self):
        _, self.deriv = self.op.linearize(self.x)
        self._aux = self.deriv(self._d)
        self._aux2 = self.op.codomain.gram(self._aux)
        self._alpha = (self._innerProd
                       / np.real(self.op.codomain.inner(self._aux, self._aux2)))
        self._s2 += -self._alpha * self._aux2
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.op.domain.gram_inv(self._rtilde)
        self._beta = (np.real(self.op.codomain.inner(self._r, self._rtilde))
                      / self._innerProd)

    def _next(self):
        while (np.sqrt(self.op.domain.inner(self._s2, self._s))
               > self.rho * self._norms0 and
               self._k <= self.cgmaxit):
            self._inner_update()
            self._x_k += self._alpha * self._d
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.x += self._x_k
        self._outer_update()


class NewtonCGFrozen(Solver):
    def __init__(self, setting, data, init, cgmaxit=50, rho=0.8):
        super().__init__()
        self.setting = setting
        self.op = setting.op
        self.data = data
        self.x = init
        _, self.deriv = self.op.linearize(self.x)
        self._n = 1
        self._outer_update()
        self.rho = rho
        self.cgmaxit = cgmaxit

    def _outer_update(self):
        if int(self._n / 10) * 10 == self._n:
            _, self.deriv = self.op.linearize(self.x)
        self._x_k = self.op.domain.zeros()
        #        self._x_k = 1j*np.zeros(np.shape(self.x))
        self.y = self.op(self.x)
        self._residual = self.data - self.y
        #        _, self.deriv=self.op.linearize(self.x)
        self._s = self._residual - self.deriv(self._x_k)
        self._s2 = self.setting.codomain.gram(self._s)
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.setting.domain.gram_inv(self._rtilde)
        self._d = self._r
        self._innerProd = self.setting.domain.inner(self._r, self._rtilde)
        self._norms0 = np.sqrt(np.real(self.setting.domain.inner(self._s2, self._s)))
        self._k = 1
        self._n += 1

    def _inner_update(self):
        _, self.deriv = self.op.linearize(self.x)
        self._aux = self.deriv(self._d)
        self._aux2 = self.setting.codomain.gram(self._aux)
        self._alpha = (self._innerProd
                       / np.real(self.setting.codomain.inner(self._aux, self._aux2)))
        self._s2 += -self._alpha * self._aux2
        self._rtilde = self.deriv.adjoint(self._s2)
        self._r = self.setting.domain.gram_inv(self._rtilde)
        self._beta = (np.real(self.setting.codomain.inner(self._r, self._rtilde))
                      / self._innerProd)

    def _next(self):
        while (
            np.sqrt(self.setting.domain.inner(self._s2, self._s)) > self.rho * self._norms0
            and self._k <= self.cgmaxit
        ):
            self._inner_update()
            self._x_k += self._alpha * self._d
            self._d = self._r + self._beta * self._d
            self._k += 1
        self.x += self._x_k
        self._outer_update()


class NewtonSemiSmooth(Solver):
    def __init__(self, setting, rhs, init, alpha, psi_minus, psi_plus):
        super().__init__()
        self.setting = setting
        self.rhs = rhs
        self.x = init
        self.alpha = alpha
        self.psi_minus = psi_minus
        self.psi_plus = psi_plus

        self.size = init.shape[0]

        self.y = self.setting.op(self.x)

        self.b = self.setting.op.adjoint(self.rhs) + self.alpha * init

        self.lam_plus = np.maximum(np.zeros(self.size), self.b - self._A(self.x))
        self.lam_minus = -np.minimum(np.zeros(self.size), self.b - self._A(self.x))

        # sets where the upper constraint and the lower constarint are active
        self.active_plus = [self.lam_plus[j] + self.alpha * (self.x[j] - self.psi_plus) > 0 for j in
                            range(self.size)]
        self.active_minus = [self.lam_minus[j] - self.alpha * (self.x[j] - self.psi_minus) > 0 for j
                             in range(self.size)]

        # complte active and inactive sets, need to be computed in each step again
        self.active = np.empty(self.size)
        self.inactive = np.empty(self.size)

    def _next(self):
        self.active = [self.active_plus[j] or self.active_minus[j] for j in range(self.size)]
        self.inactive = [self.active[j] == False for j in range(self.size)]

        # On the active sets the solution takes the values of the constraints
        self.x[self.active_plus] = self.psi_plus
        self.x[self.active_minus] = self.psi_minus

        self.lam_plus[self.inactive] = 0
        self.lam_plus[self.active_minus] = 0
        self.lam_minus[self.inactive] = 0
        self.lam_minus[self.active_plus] = 0

        # A as spla.LinearOperator constrained to inactive set
        A_inactive = spla.LinearOperator(
            (np.count_nonzero(self.inactive), np.count_nonzero(self.inactive)),
            matvec=self._A_inactive,
            dtype=float)
        # Solve system on the different sets
        self.x[self.inactive] = self._gmres(A_inactive,
                                            self.b[self.inactive] + self.lam_minus[self.inactive] -
                                            self.lam_plus[self.inactive])
        z = self._A(self.x)
        self.lam_plus[self.active_plus] = self.b[self.active_plus] + self.lam_minus[
            self.active_plus] - z[self.active_plus]
        self.lam_minus[self.active_minus] = -self.b[self.active_minus] + self.lam_plus[
            self.active_minus] + z[self.active_minus]

        # Update active and inactive sets
        self.y = self.setting.op(self.x)
        self.active_plus = [self.lam_plus[j] + self.alpha * (self.x[j] - self.psi_plus) > 0 for j in
                            range(self.size)]
        self.active_minus = [self.lam_minus[j] - self.alpha * (self.x[j] - self.psi_minus) > 0 for j
                             in range(self.size)]

    def _gmres(self, op, rhs):
        result, info = spla.gmres(op, rhs.ravel())
        if info > 0:
            self.log.warn('Gmres failed to converge')
        elif info < 0:
            self.log.warn('Illegal Gmres input or breakdown')
        return result

    def _A(self, u):
        self.y = self.setting.op(u)
        return self.alpha * u + self.setting.op.adjoint(self.y)

    def _A_inactive(self, u):
        projection = np.zeros(self.size)
        projection[self.inactive] = u
        return self._A(projection)[self.inactive]
