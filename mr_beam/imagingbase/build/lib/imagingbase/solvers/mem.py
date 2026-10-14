import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util
from regpy.functionals import Functional
from regpy.operators import Multiplication

from imagingbase.ehtim_wrapper import EhtimFunctional

class Forward_Backward_Splitting(Solver):
    def __init__(self, setting, data_fidelity, penalty, init, percent_lambda=None, proximal_pars = None):
        
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.proximal_pars = proximal_pars

        self.data_fidelity = data_fidelity
        self.penalty = penalty
        """The functional of the data fidelity term and the penalty term"""
        assert isinstance(self.data_fidelity, EhtimFunctional)
        assert isinstance(self.penalty, Functional)
        assert self.penalty.Hdomain == self.setting.Hdomain
        
        vis = self.data_fidelity.handler.unpack('vis')
        sigma = self.data_fidelity.handler.unpack('sigma')
        self.percent_lambda = percent_lambda #or self._get_percent_lambda(vis)
        
        self.x = init
        self.y = self.setting.op(self.x)
        
        self.lambd = 2 * np.max(np.abs(self.y-vis)) * self.percent_lambda
        
        self.H = Multiplication(self.setting.Hcodomain.discr, 1/sigma) * self.setting.op
        self.HtH = self.setting.Hdomain.gram_inv * self.H.adjoint * self.setting.Hcodomain.gram * self.H
        self.Lip = self.HtH.norm()
        
    def _next(self):
        self.x -= 1/self.Lip*self.setting.Hdomain.gram_inv(self.data_fidelity.gradient(self.x)) 
        self.x = self.penalty.proximal(self.x, self.lambd/self.Lip, self.proximal_pars)
        """Note: If F = alpha G, then prox_{tau, F} = prox_{alpha * tau, G}"""
        
        self.y = self.setting.op(self.x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        