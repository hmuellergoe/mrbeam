import logging

import matplotlib.pyplot as plt
import numpy as np
from regpy.solvers.irgnm import IrgnmCG

import regpy.stoprules as rules
from regpy.hilbert import L2, Sobolev
from regpy.operators.obstacles import Potential
from regpy.discrs.obstacles import StarTrigDiscr
from regpy.solvers import HilbertSpaceSetting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

op = Potential(
    domain=StarTrigDiscr(200),
    radius=1.2,
    nmeas=64,
)

exact_solution = op.domain.sample(lambda t: np.sqrt(3 * np.cos(t)**2 + 1) / 2)
exact_data = op(exact_solution)
noise = 0 * op.codomain.randn()
data = exact_data + noise

init = op.domain.sample(lambda t: 1)

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=L2)

solver = IrgnmCG(
    setting, data,
    regpar=10,
    regpar_step=0.8,
    init=init,
    cgpars=dict(
        tol=1e-4
    )
)
stoprule = (
    rules.CountIterations(100) +
    rules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(noise),
        tau=1.1
    )
)

plt.ion()
fig, axs = plt.subplots(1, 2)
axs[0].set_title('Obstacle')
axs[1].set_title('Heat flux')

for n, (reco, reco_data) in enumerate(solver.until(stoprule)):
    if n % 10 == 0:
        axs[0].clear()
        axs[0].plot(*op.domain.eval_curve(exact_solution).curve[0])
        axs[0].plot(*op.domain.eval_curve(reco).curve[0])

        axs[1].clear()
        axs[1].plot(exact_data, label='exact')
        axs[1].plot(reco_data, label='reco')
        axs[1].plot(data, label='measured')
        axs[1].legend()
        axs[1].set_ylim(ymin=0)
        plt.pause(0.5)

plt.ioff()
plt.show()
