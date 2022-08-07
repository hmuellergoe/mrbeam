import scipy.optimize as opt
import numpy as np
from imagingbase.regpy_utils import RegpySolver as Solver
from imagingbase.regpy_functionals import Functional
from regpy.operators import Identity

class Minimize():
    def __init__(self, fun, x0, args=(), method=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        self.fun = fun
        self.x0 = x0
        self.args = args
        self.method = method
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

        try:
            self.jac = self.fun.gradient
        except: 
            print("Functional has no gradient")
            self.jac = None

        try:
            self.hess = self.fun.hessian
        except: 
            print("Functional has no Hessian")
            self.hess = None

    def run(self):
        return opt.minimize(self.fun, self.x0, args=self.args, method=self.method,
                                jac=self.jac, hess=self.hess, bounds=self.bounds, constraints=self.constraints,
                                tol=self.tol, callback=self.callback, options=self.options).x

class Forward_Backward_Optimization(Solver):
    def __init__(self, setting, data_fidelity, penalty, init, zbl, regpar=1, proximal_pars=None, args=(), method=None,
                    bounds=None, constraints=(), tol=None, callback=None, options=None, op=None):
        
        super().__init__()
        self.setting = setting
        self.zbl = zbl
        self.regpar = regpar
        self.proximal_pars = proximal_pars
        
        self.op = op or Identity(self.penalty.domain)

        self.data_fidelity = data_fidelity
        self.penalty = penalty
        assert isinstance(self.data_fidelity, Functional)
        assert isinstance(self.penalty, Functional)
        assert self.op.codomain == self.setting.Hdomain.discr
        
        self.x = init
        self.y = self.setting.op(self.op(self.x))

        self.minimizer = Minimize(data_fidelity, init, args=args, method=method, bounds=bounds,
                    constraints=constraints, tol=tol, callback=callback, options=options)

    def _next(self):
        self.x = self.minimizer.run()
        self.x = self.penalty.proximal(self.op(self.x), self.regpar, self.proximal_pars)
        self.y = self.setting.op(self.x)
        
        correct = self.zbl/np.sum(self.y)
        self.y *= correct
        self.x *= correct
        
        self.minimizer.x0 = self.op.adjoint(self.x)