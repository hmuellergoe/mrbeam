import logging

import numpy as np

from regpy.solvers import HilbertSpaceSetting, Solver
from regpy.solvers.tikhonov import TikhonovCG
from regpy.stoprules import CountIterations


class IrgnmCG(Solver):
    """The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

        ||T(x_n) + T'[x_n] h - data||**2 + regpar_n * ||x_n + h - init||**2

    where `T` is a Frechet-differentiable operator, using `regpy.solvers.tikhonov.TikhonovCG`.
    `regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: `2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cgpars : dict
        Parameter dictionary passed to the inner `regpy.solvers.tikhonov.TikhonovCG` solver.
    """

    def __init__(self, setting, data, regpar, regpar_step=2 / 3, init=None, cgpars=None, cgstop=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.data = data
        """The measured data."""
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar = regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""
        self.cgstop = cgstop
        """Maximum number of iterations for inner CG solver, or None"""

    def _next(self):
        if self.cgstop is not None:
            stoprule = CountIterations(self.cgstop)
            # Disable info logging, but don't override log level for all
            # CountIterations instances.
            stoprule.log = self.log.getChild('CountIterations')
            stoprule.log.setLevel(logging.WARNING)
        else:
            stoprule = None
        self.log.info('Running Tikhonov solver.')
        step, _ = TikhonovCG(
            setting=HilbertSpaceSetting(self.deriv, self.setting.Hdomain, self.setting.Hcodomain),
            data=self.data - self.y,
            regpar=self.regpar,
            xref=self.init - self.x,
            **self.cgpars
        ).run(stoprule=stoprule)
        self.x += step
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar *= self.regpar_step
        
from regpy.operators import MatrixMultiplication
from regpy import util
from scipy.sparse.linalg import eigsh
        
class IrgnmCGPrec(Solver):
    """The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

        ||F(x_n) + F'[x_n] h - data||**2 + regpar_n * ||x_n + h - init||**2

    where `F` is a Frechet-differentiable operator, by solving in every iteration step the problem

        Minimize    ||T (M @ g) - rhs||**2 + regpar * ||M @ (g - xref)||**2
        M @ h = g

    with `regpy.solvers.tikhonov.TikhonovCG' and spectral preconditioner M.
    The spectral preconditioner M is chosen, such that:
        M @ A @ M \approx Id
    where A = (Gram_domain^(-1) T^t Gram_codomain T + regpar*Id) = T^* T + regpar Id 

    Note that the Tikhonov CG solver computes an orthonormal basis of vectors spanning the Krylov subspace of 
    the order of the number of iterations: {v_j}
    We approximate A by the operator:
    C_k: v \mapsto regpar * v +\sum_{j=1}^k <v, v_j> lambda_j v_j
    where lambda are the biggest eigenvalues of T*T.
    
    We choose: M = C_k^(-1/2) and M^(-1) = C_k^(1/2)

    It is:
    M     : v \mapsto 1/sqrt(regpar) v + \sum_{j=1}^{k} [1/sqrt(lambda_j+regpar)-1/sqrt(regpar)] <v_j, v> v_j 
    M^(-1): v \mapsto sqrt(regpar) v + \sum_{j=1}^{k} [sqrt(lambda_j+regpar) -sqrt(regpar)] <v_j, v> v_j

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: `2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cgpars : dict
        Parameter dictionary passed to the inner `regpy.solvers.tikhonov.TikhonovCG` solver.
    precpars : dict
        Parameter dictionary passed to the computation of the spectral preconditioner
    """

    def __init__(self, setting, data, regpar, regpar_step=2 / 3, init=None, cgpars=None, precpars=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.data = data
        """The measured data."""
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar = regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""
        
        self.k=0
        """Counts the number of iterations"""

        if precpars is None:
            self.krylov_order = 5
            """Order of krylov space in which the spetcral preconditioner is computed"""
            self.number_eigenvalues = 5
            """Spectral preonditioner computed only from the biggest eigenvalues """
        else: 
            self.krylov_order = precpars['krylov_order']
            self.number_eigenvalues = precpars['number_eigenvalues']

        self.krylov_basis = np.zeros((self.krylov_order, self.setting.Hdomain.discr.size))
        """Orthonormal Basis of Krylov subspace"""
        self.need_prec_update = True
        """Is an update of the preconditioner needed"""
                
    def _next(self):
        self.log.info('Running Tikhonov solver.')
        
        if self.need_prec_update:
            self.log.info('Spectral Preconditioner needs to be updated')
            step, _ = TikhonovCG(
                setting=HilbertSpaceSetting(self.deriv, self.setting.Hdomain, self.setting.Hcodomain),
                data=self.data - self.y,
                regpar=self.regpar,
                krylov_basis=self.krylov_basis,
                xref=self.init - self.x,
                **self.cgpars
            ).run()
            self.need_prec_update = False
            self._preconditioner_update()
            self.log.info('Spectral Preconditioner updated')
          
        else:
            preconditioner = MatrixMultiplication(self.M, domain=self.setting.Hdomain.discr, codomain=self.setting.Hdomain.discr)
            step, _ = TikhonovCG(
                setting=HilbertSpaceSetting(self.deriv, self.setting.Hdomain, self.setting.Hcodomain),
                data=self.data - self.y,
                regpar=self.regpar,
                xref=self.init-self.x,
                preconditioner=preconditioner,
                **self.cgpars
            ).run()
            step = self.M @ step
            
        self.x += step
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar *= self.regpar_step
        
        self.k+=1
        if (int(np.sqrt(self.k)))**2 == self.k:
            self.need_prec_update = True
                       
    def _preconditioner_update(self):
        """perform lanzcos method to calculate the preconditioner"""
        L = np.zeros((self.krylov_order, self.krylov_order))
        for i in range(0, self.krylov_order):
            L[i, :] = np.dot(self.krylov_basis, self.setting.Hdomain.gram_inv(
                self.deriv.adjoint(
                    self.setting.Hcodomain.gram(self.deriv((self.krylov_basis[i, :]))))))
        """Express T*T in Krylov_basis"""

        #TODO: Replace eigsh by Lanczos method to estimate the greatest eigenvalues
        lamb, U = eigsh(L, self.number_eigenvalues, which='LM')
        """Perform the computation of eigenvalues and eigenvectors"""

        diag_lamb = np.diag( np.sqrt(1 / (lamb + self.regpar) ) - np.sqrt(1 / self.regpar) )
        M_krylov = np.float64(U @ diag_lamb @ U.transpose())
        self.M = self.krylov_basis.transpose() @ M_krylov @ self.krylov_basis + np.sqrt(1/self.regpar) * np.identity(self.krylov_basis.shape[1])
        """Compute preconditioner"""

        diag_lamb = np.diag ( np.sqrt(lamb + self.regpar) - np.sqrt(self.regpar) )
        M_krylov = np.float64(U @ diag_lamb @ U.transpose())
        self.M_inverse = self.krylov_basis.transpose() @ M_krylov @ self.krylov_basis + np.sqrt(self.regpar) * np.identity(self.krylov_basis.shape[1]) 
        """Compute inverse preconditioner matrix"""

