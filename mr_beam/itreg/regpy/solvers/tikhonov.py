import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util

from regpy.operators import Identity


class TikhonovCG(Solver):
    """The Tikhonov method for linear inverse problems. Minimizes

        ||T x - data||**2 + regpar * ||x - xref||**2

    using a conjugate gradient method.

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The regularization parameter. Must be positive.
    tol : float, optional
        The tolerance for the residual relative to the initial at which to stop. Default is
        the machine epsilon. Iterating beyond this point produces `NaN`s.
    reltolx, reltoly : float, optional
        Relative tolerance in domain and codomain.
    krylov_basis : Compute orthonormal basis vectors of the Krylov subspaces while running CG solver
    """
    def __init__(self, setting, data, regpar, xref=None, tol=util.eps, reltolx=None, reltoly=None, krylov_basis=None, preconditioner=None):
        assert setting.op.linear

        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.regpar = regpar
        """The regularization parameter."""
        self.tol = tol
        """The tolerance."""

        # TODO Improve documentation for these two.
        self.reltolx = reltolx
        """The relative tolerance in the domain."""
        self.reltoly = reltoly
        """The relative tolerance in the codomain."""

        self.x = self.setting.op.domain.zeros()
        if self.reltolx is not None:
            self.norm_x = 0
        self.y = self.setting.op.codomain.zeros()
        if self.reltoly is not None:
            self.g_y = self.setting.op.codomain.zeros()
            self.norm_y = 0

        
        if preconditioner is None:
            self.preconditioner = Identity (self.setting.Hdomain.discr)
            self.penalty = Identity (self.setting.Hdomain.discr)
        else: 
            self.preconditioner = preconditioner
            self.penalty = self.preconditioner * self.setting.Hdomain.gram * self.preconditioner * self.setting.Hdomain.gram_inv

        self.g_res = self.preconditioner( self.setting.op.adjoint(self.setting.Hcodomain.gram(data)) )
        """The gram matrix applied to the residual."""
        if xref is not None:
            self.g_res += self.regpar *self.preconditioner( self.setting.Hdomain.gram(xref) )
        res = self.setting.Hdomain.gram_inv(self.g_res)
        """The residual."""
        self.norm_res = np.real(np.vdot(self.g_res, res))
        """The norm of the residual."""
        self.norm_res_init = self.norm_res
        """The norm of the residual in the first iteration, for `tol`."""
        self.dir = res
        """The direction of descent."""
        self.g_dir = np.copy(self.g_res)
        """The gram matrix applied to the direction of descent."""
        # TODO Improve documentation
        self.kappa = 1
        """Auxiliary parameter for estimating the relative tolerances."""

        self.krylov_basis=krylov_basis
        if self.krylov_basis is not None: 
            self.iteration_number=0
            self.krylov_basis[self.iteration_number, :] = res / np.linalg.norm(res)
        """In every iteration step of the Tikhonov solver a new orthonormal vector is computed"""


    def _next(self):
        Tdir = self.setting.op( self.preconditioner(self.dir) )
        g_Tdir = self.setting.Hcodomain.gram(Tdir)
        stepsize = self.norm_res / np.real(
            np.vdot(g_Tdir, Tdir) + self.regpar * np.vdot(self.penalty (self.g_dir), self.dir)
        )

        self.x += stepsize * self.dir
        if self.reltolx is not None:
            self.norm_x = np.real(np.vdot(self.x, self.setting.Hdomain.gram(self.x)))

        self.y += stepsize * Tdir
        if self.reltoly is not None:
            self.g_y += stepsize * g_Tdir
            self.norm_y = np.real(np.vdot(self.g_y, self.y))

        self.g_res -= stepsize * (self.preconditioner( self.setting.op.adjoint(g_Tdir) )+ self.regpar * self.penalty (self.g_dir) )
        res = self.setting.Hdomain.gram_inv(self.g_res)

        norm_res_old = self.norm_res
        self.norm_res = np.real(np.vdot(self.g_res, res))
        beta = self.norm_res / norm_res_old

        if self.krylov_basis is not None:
            self.iteration_number+=1
            if self.iteration_number < self.krylov_basis.shape[0]:
                self.krylov_basis[self.iteration_number, :] = res / np.linalg.norm(res)

        self.kappa = 1 + beta * self.kappa

        if self.krylov_basis is None or self.iteration_number > self.krylov_basis.shape[0]:
            """If Krylov subspace basis is computed, then stop the iteration only if the number of iterations exceeds the order of the Krylov space"""
            
            if (
                self.reltolx is not None and
                np.sqrt(self.norm_res / self.norm_x / self.kappa) / self.regpar
                    < self.reltolx / (1 + self.reltolx)
            ):
                return self.converge()

            if (
                self.reltoly is not None and
                np.sqrt(self.norm_res / self.norm_y / self.kappa / self.regpar)
                    < self.reltoly / (1 + self.reltoly)
            ):
                return self.converge()

            if (
                self.tol is not None and
                np.sqrt(self.norm_res / self.norm_res_init / self.kappa) < self.tol
            ):
                return self.converge()

        self.dir *= beta
        self.dir += res
        self.g_dir *= beta
        self.g_dir += self.g_res
