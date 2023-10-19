from regpy.solvers import Solver
from regpy.functionals import Functional

"""
Minimizes data_fidelity(f) with gradient descent algorithm

Parameters
----------
setting : regpy.solvers.HilbertSpaceSetting
    The setting of the forward problem.
data_fidelity : regpy.functionals.Functional
    The data fidelity term. Needs to have a gradient defined.
init : array-like
    The initial guess. 
tau : float , optional
    The stepsize
"""

class Gradient_Descent(Solver):
    def __init__(self, setting, data_fidelity, init, tau = 1):
        
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.tau = tau
        """The step size"""

        self.data_fidelity = data_fidelity
        assert isinstance(self.data_fidelity, Functional)
        
        self.x = init
        self.y = self.setting.op(self.x)
        
    def _next(self):
        self.x-=self.tau*self.setting.Hdomain.gram_inv(self.data_fidelity.gradient(self.x))       
        self.y = self.setting.op(self.x)

