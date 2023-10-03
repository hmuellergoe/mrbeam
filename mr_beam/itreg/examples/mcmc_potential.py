import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from regpy.solvers import HilbertSpaceSetting

from regpy.solvers.irgnm import IrgnmCG

from regpy.discrs.obstacles import StarTrigDiscr
from regpy.functionals import ErrorToInfinity

from regpy.operators.obstacles import Potential

from regpy.mcmc import RandomWalk, StateHistory, adaptive_stepsize
from regpy.operators.volterra import Volterra
from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid
import regpy.stoprules as rules


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

# Determine an initial estimate using Irgnm
estimate, _ = IrgnmCG(
    setting=HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=L2),
    data=data,
    regpar=10,
    regpar_step=0.8,
    init=op.domain.sample(lambda t: 1),
).run(rules.CountIterations(100))

# Compute log probability functional as negative Tikhonov functional with Sobolev regularizer
regpar = 1e-2
temperature = 1e-3
prior = regpar * Sobolev(op.domain).norm_functional
# ErrorToInfinity turns errors raised by the forward operator (when called with an invalid
# obstacle) into inf to reject the proposal
likelihood = ErrorToInfinity(L2(op.codomain).norm_functional * (op - data))
logpdf = -(likelihood + prior) / temperature

# Initialize MCMC sampler
init = op.domain.ones()
sampler = RandomWalk(
    logpdf=logpdf,
    stepsize=0.01,
    state=RandomWalk.State(pos=estimate),
    # Increase stepsize when accepting, decrease when rejecting
    stepsize_rule=partial(adaptive_stepsize, stepsize_factor=1.05)
)

# Glorified list keeping the last 1e4 samples. We plan to implement history managers that store
# samples on disk for larger problems.
hist = StateHistory(maxlen=1e4)

# The simples way to run the sampler is to use
#
#    sampler.run(niter=1e5, callback=hist.add)
#
# `run()` is a thin wrapper around iterating over the sampler and handing over each state to the
# callback. However, the StateHistory will only keep `maxlen` states. The following keeps the
# logprobs completely, to allow plotting convergence.

logprobs = []
for n, (state, accepted) in zip(range(int(5e4)), sampler):
    hist.add(state, accepted)
    if accepted:
        logprobs.append(state.logprob)
    if n > 0 and n % 1000 == 0:
        logging.info(
            'MCMC step {}. Acceptance rate: {}'.format(n, hist.acceptance_rate)
        )

logging.info('MCMC finished. Acceptance rate: {} / {} = {}'.format(
        hist.accepted, hist.total, hist.acceptance_rate
))

samples = hist.samples()
mean = np.mean(samples, axis=0)
std = np.std(samples, axis=0)

# If we had not kept logprobs manually, this could be used to get them:
#
#     logprobs = hist.logprobs()

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0, 0].set_title('logprob')
axes[0, 0].plot(logprobs)
axes[0, 0].plot(logpdf(exact_solution) * np.ones(len(logprobs)))

axes[0, 1].set_title('hist mean +/- 1 sigma')
axes[0, 1].plot(*op.domain.eval_curve(exact_solution).curve[0])
axes[0, 1].plot(*op.domain.eval_curve(mean).curve[0])
axes[0, 1].plot(*op.domain.eval_curve(mean - std).curve[0])
axes[0, 1].plot(*op.domain.eval_curve(mean + std).curve[0])

axes[1, 0].set_title('sample reco')
axes[1, 0].plot(*op.domain.eval_curve(exact_solution).curve[0])
axes[1, 0].plot(*op.domain.eval_curve(samples[-1]).curve[0])

axes[1, 1].set_title('sample data')
axes[1, 1].plot(exact_data)
axes[1, 1].plot(op(samples[-1]))

plt.show()
