'''
The volterra operator inversion with generalized FISTA.
TV is used as default for penalty term. For the use of other penalty terms, see
volterra_forward_backward_splitting.py
'''

import logging

import matplotlib.pyplot as plt
import numpy as np

import regpy.stoprules as rules
from regpy.operators.volterra import Volterra
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.fista import FISTA
from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid
from regpy.functionals import HilbertNorm, TV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

grid = UniformGrid(np.linspace(0, 2 * np.pi, 200))
op = Volterra(grid, exponent=3)

"""Impulsive Noise"""
sigma = 0.01*np.ones(grid.coords.shape[1])
sigma[100:110] = 0.5

exact_solution = np.sin(grid.coords[0])
exact_data = op(exact_solution)
noise = sigma * op.domain.randn()
data = exact_data + noise
init = op.domain.ones()

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=L2)

data_fidelity_operator = op - data
data_fidelity = HilbertNorm(setting.Hcodomain) * data_fidelity_operator
"""The penalty term: 1/2 * ||f||_{TV}^2"""
penalty = TV(setting.Hdomain.discr, Hdomain=setting.Hdomain)

proximal_pars = {
        'stepsize' : 0.001,
        'maxiter' : 100
        }
"""Parameters for the inner computation of the proximal operator with the Chambolle algorithm"""

tau = 0.01
alpha = 0.01

solver = FISTA(setting, data_fidelity, penalty, init, tau = tau, regpar = alpha, proximal_pars=proximal_pars)
stoprule = (
    # Method is slow, so need to use large number of iterations
    rules.CountIterations(max_iterations=100000) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
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