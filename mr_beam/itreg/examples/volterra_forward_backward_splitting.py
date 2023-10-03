'''
The volterra operator inversion with forward-backward-splitting.
TV is used as default for penalty term.
'''

import logging

import matplotlib.pyplot as plt
import numpy as np

import regpy.stoprules as rules
from regpy.operators.volterra import Volterra
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.forward_backward_splitting import Forward_Backward_Splitting
from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid
from regpy.functionals import HilbertNorm, L1, TV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

grid = UniformGrid(np.linspace(0, 2 * np.pi, 200))
op = Volterra(grid, exponent=3)

exact_solution = np.cos(grid.coords[0])
exact_data = op(exact_solution)
noise = 0.3 * op.domain.randn()
data = exact_data + noise
init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=L2)

data_fidelity_operator = op - data
data_fidelity = HilbertNorm(setting.Hcodomain) * data_fidelity_operator
"""The data fidelity term: 1/2*||op(f)-data||^2"""
"""Uncomment to use L1 norm as penalty term instead"""
#penalty = L1(setting.Hdomain.discr)
"""The penalty term: 1/2 * ||f||_{TV}^2"""
penalty = TV(setting.Hdomain.discr, Hdomain=setting.Hdomain)

proximal_pars = {
        'stepsize' : 0.0001,
        'maxiter' : 1000
        }
"""Parameters for the inner computation of the proximal operator with the Chambolle algorithm"""

tau = 0.01
alpha = 0.5

solver = Forward_Backward_Splitting(setting, data_fidelity, penalty, init, tau = tau, regpar = alpha, proximal_pars=proximal_pars)
stoprule = (
    # Method is slow, so need to use large number of iterations
    rules.CountIterations(max_iterations=2000)+
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.2
    )
)

reco, reco_data = solver.run(stoprule)

plt.plot(grid.coords[0], exact_solution.T, label='exact solution')
plt.plot(grid.coords[0], reco, label='reco')
plt.plot(grid.coords[0], exact_data, label='exact data')
plt.plot(grid.coords[0], data, label='data')
plt.plot(grid.coords[0], reco_data, label='reco data')
plt.legend()
plt.show()
