import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from regpy.mcmc import RandomWalk, StateHistory, adaptive_stepsize
from regpy.operators.volterra import Volterra
from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

op = Volterra(UniformGrid((0, 2 * np.pi, 200)))

# Simulate data
exact_solution = np.sin(op.domain.coords[0])
exact_data = op(exact_solution)
data = exact_data + 0.03 * op.domain.randn()

# Compute log probability functional as negative Tikhonov functional with Sobolev regularizer

regpar = 1e-1
temperature = 1e-3

prior = regpar * Sobolev(op.domain).norm_functional
likelihood = L2(op.codomain).norm_functional * (op - data)
logpdf = -(likelihood + prior) / temperature

# Initialize MCMC sampler

init = op.domain.ones()
sampler = RandomWalk(
    logpdf=logpdf,
    stepsize=0.1,
    state=RandomWalk.State(pos=init),
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
axes[0, 1].plot(exact_solution)
axes[0, 1].plot(mean)
axes[0, 1].plot(mean - std)
axes[0, 1].plot(mean + std)

axes[1, 0].set_title('sample reco')
axes[1, 0].plot(exact_solution)
axes[1, 0].plot(samples[-1])

axes[1, 1].set_title('sample data')
axes[1, 1].plot(exact_data)
axes[1, 1].plot(op(samples[-1]))

plt.show()
