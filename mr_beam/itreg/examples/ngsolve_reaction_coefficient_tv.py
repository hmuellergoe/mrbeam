# Run this file in IPython like
#     import netgen.gui
#     %run path/to/this/file
# to get graphical output.

import logging
import ngsolve as ngs
from ngsolve.meshes import MakeQuadMesh
import numpy as np

import regpy.stoprules as rules
from regpy.operators.ngsolve import Coefficient
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.forward_backward_splitting import Forward_Backward_Splitting
from regpy.hilbert import L2, Sobolev
from regpy.discrs.ngsolve import NgsSpace
from regpy.functionals import HilbertNorm, TV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

meshsize_domain = 20
meshsize_codomain = 20

mesh = MakeQuadMesh(meshsize_domain, meshsize_domain)
fes_domain = ngs.H1(mesh, order=1)
domain = NgsSpace(fes_domain)

mesh = MakeQuadMesh(meshsize_codomain, meshsize_codomain)
bdr = "left|top|right|bottom"
fes_codomain = ngs.H1(mesh, order=3, dirichlet=bdr)
codomain = NgsSpace(fes_codomain, bdr=bdr)

rhs = 1 * ngs.sin(ngs.x) * ngs.sin(ngs.y)
op = Coefficient(
    domain, rhs, codomain=codomain, bc = 0.1, diffusion=False,
    reaction=True
)

exact_solution_coeff = 1+0.8*ngs.sin(2*np.pi*ngs.x) * ngs.sin(2*np.pi*ngs.y)
exact_solution = domain.from_ngs( exact_solution_coeff )
exact_data = op(exact_solution)

noise = 0.0005 * codomain.randn()

data = exact_data+noise

init = domain.from_ngs ( 1 )
init_data = op(init)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=Sobolev)

data_fidelity_operator = op - data
data_fidelity = HilbertNorm(setting.Hcodomain) * data_fidelity_operator
"""The penalty term: 1/2 * ||f||_{TV}^2"""
penalty = TV(setting.Hdomain.discr)

proximal_pars = {
        'stepsize' : 0.1,
        'maxiter' : 10
        }
"""Parameters for the inner computation of the proximal operator with the Chambolle algorithm"""

tau = 10
alpha = 5*10**(-6)

solver = Forward_Backward_Splitting(setting, data_fidelity, penalty, init, tau = tau, regpar = alpha, proximal_pars=proximal_pars)
stoprule = (
        rules.CountIterations(500) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

reco, reco_data = solver.run(stoprule)

domain.draw(exact_solution, "exact")

# Draw reconstructed solution
domain.draw(reco, "reco")

# Draw data space
codomain.draw(data, "data")
codomain.draw(reco_data, "reco_data")