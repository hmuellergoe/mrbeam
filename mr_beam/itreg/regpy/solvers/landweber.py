from regpy.solvers import Solver

import logging
import numpy as np


class Landweber(Solver):
    """The Landweber method. Solves the potentially non-linear, ill-posed equation

        T(x) = rhs,

    where `T` is a Frechet-differentiable operator, by gradient descent for the residual

        ||T(x) - rhs||**2,

    where `||.||` is the Hilbert space norm in the codomain, and gradients are computed with
    respect to the Hilbert space structure on the domain.

    The number of iterations is effectively the regularization parameter and needs to be picked
    carefully.

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    rhs : array-like
        The right hand side.
    init : array-like
        The initial guess.
    stepsize : float, optional
        The step length; must be chosen not too large. If omitted, it is guessed from the norm of
        the derivative at the initial guess.
    """

    def __init__(self, setting, rhs, init, stepsize=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.rhs = rhs
        """The right hand side."""
        self.x = init
        self.y, deriv = self.setting.op.linearize(self.x)
        self.deriv = deriv
        """The derivative at the current iterate."""
        self.stepsize = stepsize or 1 / self.deriv.norm()**2
        """The stepsize."""

    def _next(self):
        residual = self.y - self.rhs
        gy_residual = self.setting.Hcodomain.gram(residual)
        self.x -= self.stepsize * self.setting.Hdomain.gram_inv(self.deriv.adjoint(gy_residual))
        self.y, self.deriv = self.setting.op.linearize(self.x)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(residual, gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))
