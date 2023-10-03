# Run this file in IPython like
#     import netgen.gui
#     %run path/to/this/file
# to get graphical output.

import logging
import ngsolve as ngs
import numpy as np
from netgen.geom2d import SplineGeometry

import regpy.stoprules as rules
from regpy.operators.ngsolve import EIT, ProjectToBoundary
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import Sobolev, SobolevBoundary
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

geo = SplineGeometry()
bc = "cyc"
geo.AddCircle((0, 0), r=1, bc=bc, maxh=0.05)
mesh = ngs.Mesh(geo.GenerateMesh())

fes_domain = ngs.H1(mesh, order=1)
domain = NgsSpace(fes_domain)

fes_codomain = ngs.H1(mesh, order=4)
codomain = NgsSpace(fes_codomain, bdr=bc)**49

g = [0.01, 0.01*ngs.sin(np.pi*ngs.x), 0.01*ngs.sin(np.pi*ngs.y), 0.01*ngs.cos(np.pi*ngs.x), 0.01*ngs.cos(np.pi*ngs.y),
0.01*ngs.sin(np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.sin(np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)),
 0.01*ngs.sin(2*np.pi*ngs.x), 0.01*ngs.sin(2*np.pi*ngs.y), 0.01*ngs.cos(2*np.pi*ngs.x), 0.01*ngs.cos(2*np.pi*ngs.y),
0.01*ngs.sin(2*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.sin(2*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(2*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(2*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)),
 0.01*ngs.sin(3*np.pi*ngs.x), 0.01*ngs.sin(3*np.pi*ngs.y), 0.01*ngs.cos(3*np.pi*ngs.x), 0.01*ngs.cos(3*np.pi*ngs.y),
0.01*ngs.sin(3*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.sin(3*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(3*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(3*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)),
 0.01*ngs.sin(4*np.pi*ngs.x), 0.01*ngs.sin(4*np.pi*ngs.y), 0.01*ngs.cos(4*np.pi*ngs.x), 0.01*ngs.cos(4*np.pi*ngs.y),
0.01*ngs.sin(4*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.sin(4*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(4*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(4*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)), 
 0.01*ngs.sin(5*np.pi*ngs.x), 0.01*ngs.sin(5*np.pi*ngs.y), 0.01*ngs.cos(5*np.pi*ngs.x), 0.01*ngs.cos(5*np.pi*ngs.y),
0.01*ngs.sin(5*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.sin(5*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(5*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(5*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)),
 0.01*ngs.sin(6*np.pi*ngs.x), 0.01*ngs.sin(6*np.pi*ngs.y), 0.01*ngs.cos(6*np.pi*ngs.x), 0.01*ngs.cos(6*np.pi*ngs.y),
0.01*ngs.sin(6*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.sin(6*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(6*np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)), 0.01*ngs.cos(6*np.pi*(ngs.x-ngs.y)/ngs.sqrt(2))]
  
eit = EIT(domain, g, codomain=codomain, alpha=10**(-2))
proj = ProjectToBoundary(codomain)
op = proj * eit

exact_solution_coeff = 1+0.5*ngs.exp(-2*(ngs.x)**2-2*(ngs.y-0.9)**2)+0.5*ngs.exp(-2*(ngs.x-0.9)**2-2*(ngs.y)**2)
exact_solution = domain.from_ngs( exact_solution_coeff )
exact_data = op(exact_solution)

noise = proj( 0*0.00005*codomain.randn() )

data = exact_data+noise

init = domain.from_ngs(1)

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=SobolevBoundary)

#Discrepancy Principle usually stops very early
landweber = Landweber(setting, data, init, stepsize=1)
stoprule = (
        rules.CountIterations(500) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.2)
)

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, fes_domain.mesh, "exact")

# Draw reconstructed solution
domain.draw(reco, "reco")

# Draw data space
reco_data = codomain.split(reco_data)
codomain.summands[5].draw(reco_data[5], "reco_data")
codomain.summands[5].draw(codomain.split(data)[5], "data")