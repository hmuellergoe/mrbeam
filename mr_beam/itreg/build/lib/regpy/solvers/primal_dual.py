import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util
from regpy.functionals import Functional

"""The Primal-dual hybrid gradient (PDHG) or Chambolle-Pock Algorithm
    For theta==0 this is the Arrow-Hurwicz-Uzawa algorithm.

    Solves the minimization problem: data_fidelity(Tf)+regpar*penalty(f)
    by solving the saddle-point problem: inf_f sup_p [ <Tf,p>+regpar*penalty(f)-Fenchel conjugate of data_fidelity(p) ]

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem. The operator needs to be linear.
    data_fidelity_conjugate : regpy.functionals.Functional
        The Fenchel conjugate of the data fidelity functional. Needs to have a prox-operator defined.
    penalty : regpy.functionals.Functional
        The penalty term. Needs to have a prox-operator defined.
    init_domain : array_like
        The initial guess "f".
    init_codomain : array-like
        The initial guess "p". 
    tau : float , optional
        The parameter to compute the proximal operator of the penalty term. Must be positive. Stepsize of the primal step.
    sigma : float , optional
        The parameter to compute the proximal operator of the data-fidelity term. Must be positive. Stepsize of the dual step.
    regpar : float, optional
        The regularization parameter. Must be positive.
    theta : float, optional
        Relaxation parameter. For theta==0 PDHG is the Arrow-Hurwicz-Uzawa algorithm.
    proximal_pars_data_fidelity_conjugate : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the data fidelity functional.
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator of the penalty functional.
    """
class PDHG(Solver):
    def __init__(self,  setting, data_fidelity_conjugate, penalty, init_domain, init_codomain, tau = 1, sigma = 1, regpar = 1, theta= 0, proximal_pars_data_fidelity_conjugate = None, proximal_pars_penalty = None):
        super().__init__()
        self.setting = setting
        assert self.setting.op.linear
        self.data_fidelity_conjugate = data_fidelity_conjugate
        self.penalty = penalty
        assert isinstance(self.data_fidelity_conjugate, Functional)
        assert isinstance(self.penalty, Functional)
        assert self.penalty.Hdomain == self.setting.Hdomain

        self.x = init_domain
        self.x_old = self.x
        self.y = self.setting.op(self.x)
        self.p = init_codomain

        self.tau = tau
        self.sigma = sigma
        self.regpar = regpar
        self.theta = theta
        self.proximal_pars_data_fidelity_conjugate = proximal_pars_data_fidelity_conjugate
        self.proximal_pars_penalty = proximal_pars_penalty

    def _next(self):
        primal_step = self.x - self.tau * self.setting.Hdomain.gram_inv(self.setting.op.adjoint(self.setting.Hcodomain.gram(self.p)))
        self.x = self.penalty.proximal(primal_step, self.regpar * self.tau, self.proximal_pars_penalty)
        dual_step = self.p + self.sigma * self.setting.op( self.x+self.theta*(self.x-self.x_old) )
        self.p = self.data_fidelity_conjugate.proximal(dual_step, self.sigma, self.proximal_pars_data_fidelity_conjugate)
        self.x_old = self.x
        self.y = self.setting.op(self.x)

"""The Douglas-Rashford Splitting Algorithm

    Minimizes Data_fidelity(f)+regpar*penalty(f)
    Parameters
        ----------
        setting : regpy.solvers.HilbertSpaceSetting
            The setting of the forward problem. The operator needs to be linear.
        data_fidelity : regpy.functionals.Functional
            The data fidelity functional. Needs to have a prox-operator defined.
        penalty : regpy.functionals.Functional
            The penalty term. Needs to have a prox-operator defined.
        init_h : array_like
            The initial guess "f".
        tau : float , optional
            The parameter to compute the proximal operator of the penalty term. Must be positive.
        regpar : float, optional
            The regularization parameter. Must be positive.
        proximal_pars_data_fidelity : dict, optional
            Parameter dictionary passed to the computation of the prox-operator of the data fidelity functional.
        proximal_pars_penalty : dict, optional
            Parameter dictionary passed to the computation of the prox-operator of the penalty functional.
"""
class Douglas_Rashford(Solver):
    def __init__(self,  setting, data_fidelity, penalty, init_h, tau = 1, regpar = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None):
        super().__init__()
        self.setting = setting
        self.data_fidelity = data_fidelity
        self.penalty = penalty
        assert isinstance(self.data_fidelity, Functional)
        assert isinstance(self.penalty, Functional)
        assert self.data_fidelity.Hdomain == self.setting.Hcodomain
        assert self.penalty.Hdomain == self.setting.Hdomain

        self.h = init_h

        self.tau = tau
        self.regpar = regpar
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        self.proximal_pars_penalty = proximal_pars_penalty

        self.x = self.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.setting.op(self.x)

    def _next(self):
        self.h += self.data_fidelity.proximal(2*self.x-self.h, self.tau, self.proximal_pars_data_fidelity) - self.x
        self.x = self.penalty.proximal(self.h, self.tau*self.regpar, self.proximal_pars_penalty)
        self.y = self.setting.op(self.x)