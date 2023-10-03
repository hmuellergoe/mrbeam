import logging
import matplotlib.pyplot as plt
import numpy as np
import ngsolve as ngs
from ngsolve.meshes import Make1DMesh

import regpy.stoprules as rules
from regpy.operators.ngsolve import Coefficient
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import L2
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

meshsize_domain = 100
meshsize_codomain = 100

mesh = Make1DMesh(meshsize_domain)
fes_domain = ngs.H1(mesh, order=1)
domain = NgsSpace(fes_domain)

mesh = Make1DMesh(meshsize_codomain)
bdr = "left|right"
fes_codomain = ngs.H1(mesh, order=2, dirichlet=bdr)
codomain = NgsSpace(fes_codomain, bdr)

rhs = 10 * ngs.x ** 2
op = Coefficient(domain, codomain=codomain, rhs=rhs, bc=1+0.1*ngs.x, diffusion=False, reaction=True)

N_domain = op.fes_domain.ndof
N_codomain = op.fes_codomain.ndof

#exact_solution and exact_data store the coefficient vector
#of the exact solution and the exact data grid functions
exact_solution = domain.from_ngs( 1 + ngs.sin(2*np.pi*ngs.x) )
exact_data = op(exact_solution)

noise = 0.01*codomain.randn()
data = exact_data + noise

init = domain.from_ngs( 1 )

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

landweber = Landweber(setting, data, init, stepsize=1)
stoprule = (
        rules.CountIterations(10000) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1))

reco, reco_data = landweber.run(stoprule)

#For graphical output in matplotlib
gfu_domain = ngs.GridFunction(op.fes_domain)
func_domain = np.zeros(N_domain)
def plot_domain(vec, label):
    for i in range(N_domain):
        gfu_domain.vec[i] = vec[i]
    Symfunc_domain = ngs.CoefficientFunction(gfu_domain)
    for i in range(0, N_domain):
        mip = op.fes_domain.mesh(i / N_domain)
        func_domain[i] = Symfunc_domain(mip)
    plt.plot(func_domain, label=label)
    
plot_domain(reco, 'reco')
plot_domain(exact_solution, 'exact')
plt.legend()
plt.show()

#For graphical output in matplotlib
gfu_codomain = ngs.GridFunction(op.fes_codomain)
func_codomain = np.zeros(N_codomain)
def plot_codomain(vec, label):
    for i in range(N_codomain):
        gfu_codomain.vec[i] = vec[i]
    Symfunc_codomain = ngs.CoefficientFunction(gfu_codomain)
    for i in range(0, N_codomain):
        mip = op.fes_codomain.mesh(i / N_codomain)
        func_codomain[i] = Symfunc_codomain(mip)
    plt.plot(func_codomain, label=label)
    
plot_codomain(reco_data, 'reco_data')
plot_codomain(exact_data, 'exact')
plot_codomain(data, 'data')
plt.legend()
plt.show()