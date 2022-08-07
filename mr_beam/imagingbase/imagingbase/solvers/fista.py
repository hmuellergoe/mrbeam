import logging
import numpy as np

from imagingbase.regpy_utils import RegpySolver as Solver
from regpy import util
from imagingbase.regpy_functionals import Functional

"""
The generalized FISTA algorithm for minimization of regpar * G+H (where G, H: Hdomain -> R are the penalty term and the data fidelity term respectively).
We assume:
    -> G, H are convex
    -> grad H is L-Lipschitz continuous
    -> G is mu_G-convex with mu_G >= 0 
    -> H is mu_H-convex with mu_H >= 0 

Parameters:
-----------
setting : regpy.solvers.HilbertSpaceSetting
    The setting of the forward problem.
data_fidelity : regpy.functionals.Functional
    The data fidelity term. Needs to have a gradient defined. Matches H.
penalty : regpy.functionals.Functional
    The penalty term. Needs to have a prox-operator defined. Matches G.
init : array-like
    The initial guess
tau : float, optional 
    step size of minimization procedure. Needs to be in (0, 1/L) where grad H is assumed to be L-Lipschitz 
regpar : float, optional
    The regularization parameter
mu_data_fidelity : float, optional
    The convexity constant of the data fidelity term. Matches mu_H.
mu_penalty : float, optional
    The convexity constant of the penalty term. Matches mu_H
proximal_pars : dict, optional
    Parameter dictionary passed to the computation of the prox-operator for the penalty term
"""

class FISTA(Solver):
    def __init__(self, setting, data_fidelity, penalty, init, tau = 1, regpar = 1, mu_data_fidelity = 1, mu_penalty = 1, proximal_pars=None):
        super().__init__()
        self.setting = setting
        self.data_fidelity = data_fidelity
        self.penalty = penalty
        assert isinstance(self.data_fidelity, Functional)
        assert isinstance(self.penalty, Functional)
        assert self.penalty.Hdomain.discr == self.setting.Hdomain.discr

        self.x = init
        self.y = self.setting.op(self.x)

        self.tau = tau
        self.regpar = regpar
        self.mu_data_fidelity = mu_data_fidelity
        self.mu_penalty = mu_penalty
        self.proximal_pars = proximal_pars

        self.t = 0
        self.t_old = 0
        self.mu = self.mu_data_fidelity+self.mu_penalty

        self.x_old = self.x
        self.q = (self.tau * self.mu) / (1+self.tau*self.mu_penalty)

    def _next(self):
        if self.mu == 0:
            self.t = (1 + np.sqrt(1+4*self.t_old**2))/2
            beta = (self.t_old-1) / self.t
        else: 
            self.t = (1-self.q*self.t_old**2+np.sqrt((1-self.q*self.t_old**2)**2+4*self.t_old**2))/2
            beta = (self.t_old-1)/self.t * (1+self.tau*self.mu_penalty-self.t*self.tau*self.mu)/(1-self.tau*self.mu_data_fidelity)

        h = self.x+beta*(self.x-self.x_old)

        self.x_old = self.x
        self.t_old = self.t

        self.x = self.penalty.proximal(h-self.tau*self.setting.Hdomain.gram_inv(self.data_fidelity.gradient(h)), self.tau * self.regpar, self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        self.y = self.setting.op(self.x)
