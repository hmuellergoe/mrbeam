import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util
from regpy.functionals import Functional

"""
Minimizes data_fidelity(f)+regpar*penalty(f) with forward backward splitting

Parameters
----------
setting : regpy.solvers.HilbertSpaceSetting
    The setting of the forward problem.
data_fidelity : regpy.functionals.Functional
    The data fidelity term. Needs to have a prox-operator defined. Matches S.
penalty : regpy.functionals.Functional
    The penalty term. Needs to have a prox-operator defined. Matches R.
init : array-like
    The initial guess. 
tau : float , optional
    The parameter to compute the proximal operator of the penalty term. Must be positive.
regpar : float, optional
    The regularization parameter. Must be positive.
proximal_pars: dict, optional
    Parameter dictionary passed to the computation of the prox-operator.
"""

class Forward_Backward_Splitting(Solver):
    def __init__(self, setting, data_fidelity, penalty, init, tau = 1, regpar = 1, proximal_pars = None):
        
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.regpar = regpar
        """The regularization parameter."""
        self.tau = tau
        """The proximal operator parameter"""
        self.proximal_pars = proximal_pars

        self.data_fidelity = data_fidelity
        self.penalty = penalty
        """The functional of the data fidelity term and the penalty term"""
        assert isinstance(self.data_fidelity, Functional)
        assert isinstance(self.penalty, Functional)
        assert self.penalty.Hdomain == self.setting.Hdomain
        
        self.x = init
        self.y = self.setting.op(self.x)
        
    def _next(self):
        self.x-=self.tau*self.setting.Hdomain.gram_inv(self.data_fidelity.gradient(self.x)) 
        self.x = self.penalty.proximal(self.x, self.regpar*self.tau, self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        
        self.y = self.setting.op(self.x)
        
