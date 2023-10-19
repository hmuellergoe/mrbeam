"""Markov Chain Monte Carlo samplers.
"""

from collections import deque
from copy import copy
from itertools import islice

import numpy as np

from regpy.util import classlogger


class State:
    """The current state of a Metropolis-Hastings sampler. By default, the state consists of the
    current position and the log probability at the current position, but samplers can derive from
    this class to add more state attributes.

    Subclasses can override the `_complete(self, logpdf)` method, which gets handed the log
    probabilty density as a `regpy.functionals.Functional` instance and should fill missing
    attributes ("missing" means `not hasattr(self, attr)`). The default is to use a zero vector
    as position and compute the `logprob`.

    Parameters
    ----------
    **kwargs : dict
        Can be used to set arbitrary attributes.
    """

    __slots__ = 'pos', 'logprob'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def complete(self, logpdf):
        """Return a completed (shallow) copy of `self`.

        Parameters
        ----------
        logpdf : regpy.functionals.Functional
            The log probability density.

        Returns
        -------
        State
            The completed copy.
        """
        result = copy(self)
        result._complete(logpdf)
        return result

    def update(self, **kwargs):
        """Return a copy of self with attributes set accoding to the arguments.

        Parameters
        ----------
        **kwargs : dict
            The attributes and values to update.

        Returns
        -------
        State
            The updated copy.
        """
        result = copy(self)
        for k, v in kwargs.items():
            setattr(result, k, v)
        return result

    def _complete(self, logpdf):
        if not hasattr(self, 'pos'):
            self.pos = logpdf.domain.zeros()
        if not hasattr(self, 'logprob'):
            self.logprob = logpdf(self.pos)


class MetropolisHastings:
    """Abstract base class for Metropolis-Hastings samplers.

    Samplers are run by repeatedly calling the `next` method, or by using the instance as an
    iterator, which yields each return value of `next`.

    A single step consists of a proposal and an acceptance step. The acceptance check is fixed,
    but subclasses should override the `_propose(self, state)` method to generate a proposal
    starting at `state`. A new state instance should be returned, the argument should be left
    unmodified.

    Subclasses can override the `_update(self, state, accepted)` method, which will be called from
    `next` with the same values `next` returns. The default implementation updates the current state
    if the proposal is accepted, but subclasses may require more logic (like stepsize control).

    Parameters
    ----------
    logpdf : regpy.functionals.Functional
        The logarithm of the probability distribution to sample.
    state : State, optional
        The initial state. If omitted, an empty one will be generated. The state will be completed
        using the `State.complete` method, so to give only an initial guess,

            MetropolisHastings(logpdf, MetropolisHastings.State(pos=init))

        can be used (replacing MetropolisHastings by a non-abstract subclass).
    """

    State = State
    """The state class for this sampler. The base class uses `State`, but subclasses can override
    this."""

    log = classlogger

    def __init__(self, logpdf, state=None):
        self.logpdf = logpdf
        if state is not None:
            state = state.complete(logpdf)
        else:
            state = self.State().complete(logpdf)
        self.state = state
        """The current state."""

    def next(self):
        """Produce a new proposal state, and update the current state if the proposal is accepted.

        Returns
        -------
        State, bool
            The state and whether it was accepted.
        """
        state = self._propose(self.state).complete(self.logpdf)
        accepted = (np.log(np.random.rand()) < state.logprob - self.state.logprob)
        self._update(state, accepted)
        return state, accepted

    def _update(self, state, accepted):
        if accepted:
            self.state = state

    def _propose(self, state):
        raise NotImplementedError

    def __iter__(self):
        while True:
            yield self.next()

    def run(self, niter, callback=None):
        """Convenience method to run the sampler a fixed number of times.

        Parameters
        ----------
        niter : int
            Number of iterations.
        callback : callable, optional
            If given, will be called with the return value of `next` on each iteration.
            E.g. the `StateHistory` class provides a callback that memorizes a fixed number of
            iterations.
        """
        # TODO some convenience logging
        for state, accepted in islice(self, int(niter)):
            if callback is not None:
                callback(state, accepted)


class RandomWalk(MetropolisHastings):
    """Metropolis Hastings sampler that Gaussian proposal distribution.

    Parameters
    ----------
    logpdf : regpy.functionals.Functional
        The log probability distribution to sample.
    stepsize : float
        The proposal stepsize, i.e. the standard deviation of the proposal distribution.
    state : State, optional
        The initial state.
    stepsize_rule : callable, optional
        If given, will be called in every iteration with parameters:

            (current stepsize, current state, proposed state, accepted)

        and should return a new stepsize.
    """

    def __init__(self, logpdf, stepsize, state=None, stepsize_rule=None):
        super().__init__(logpdf, state=state)
        self.stepsize = float(stepsize)
        self.stepsize_rule = stepsize_rule

    def _propose(self, state):
        return self.State(
            pos=state.pos + self.stepsize * self.logpdf.domain.randn()
        ).complete(self.logpdf)

    def _update(self, state, accepted):
        if self.stepsize_rule is not None:
            self.stepsize = self.stepsize_rule(self.stepsize, self.state, state, accepted)
        super()._update(state, accepted)


def fixed_stepsize(stepsize, state, proposed, accepted):
    """Stepsize rule to be used in `RandomWalk` samplers that does not change the stepsize.
    Using it is basically equivalent to omitting the `stepsize` argument to `RandomWalk`.
    """
    return stepsize


def adaptive_stepsize(stepsize, state, proposed, accepted, stepsize_factor):
    """Stepsize rule to be used in `RandomWalk` samplers. If the proposal is accepted, the stepsize
    will be multiplied by a factor, otherwise divided by the same factor. Can be passed to a
    `RandomWalk` like

        RandomWalk(
            ...,
            stepsize_rule=functools.partial(
                adaptive_stepsize,
                stepsize_factor=1.1
            )
        )
    """
    stepsize *= stepsize_factor if accepted else 1 / stepsize_factor
    return stepsize


class HamiltonState(State):
    """State class for `HamiltonMonteCarlo` samplers.

    In addition to the `pos` and `logprob` attributes inherited from the base class, this
    contains a `grad` attribute for the logpdf gradient at the current point, and a `momenta`
    attribute for the momenta.

    Completion initializes the position and momenta to zero, and computes the log probability and
    gradient at the current point.
    """

    __slots__ = 'momenta', 'grad'

    def _complete(self, logpdf):
        if not hasattr(self, 'pos'):
            self.pos = logpdf.domain.zeros()
        if not hasattr(self, 'momenta'):
            self.momenta = logpdf.domain.zeros()
        if not hasattr(self, 'logprob'):
            # We just assume grad is not set either for simplicity.
            self.logprob, self.grad = logpdf.linearize(self.pos)
        elif not hasattr(self, 'grad'):
            self.grad = logpdf.gradient(self.pos)


def leapfrog(logpdf, state, stepsize, nsteps=10):
    """Leapfrog integrator for Hamilton equations, for use with `HamiltonianMonteCarlo`.

    Parameters
    ----------
    logpdf : regpy.functionals.Functional
        The logarithmic probability distribution.
    state : HamiltonState
        The current state.
    stepsize : float
        The time stepsize.
    nsteps : int, optional
        The number of steps. Default: 10.

    Returns
    -------
    HamiltonState
        The new state.
    """

    state = state.complete(logpdf)
    pos = state.pos.copy()
    momenta = state.momenta.copy()

    momenta += 0.5 * stepsize * state.grad

    for _ in range(nsteps):
        pos += stepsize * momenta
        momenta += stepsize * logpdf.gradient(pos)

    pos += stepsize * momenta
    logprob, grad = logpdf.linearize(pos)
    momenta += 0.5 * stepsize * grad

    return HamiltonState(pos=pos, logprob=logprob, momenta=momenta, grad=grad)


class HamiltonianMonteCarlo(RandomWalk):
    """Random walk where the propsal consists in evolving the Hamilton equations for the Hamiltonian

        H(x, p) = |p|**2 / 2 - logpdf(x),

    where `p` are auxiliary momentum variables, for a fixed amount of time, with initial conditions
    given by the `x = current position` and `p = random standard normal`.

    Parameters
    ----------
    logpdf : regpy.functionals.Functional
        The log probability distribution.
    stepsize : float
        The stepsize.
    state : HamiltonState, optional
        The initial state. The current momenta are ignored.
    stepsize_rule : callable, optional
        The stepsize rule. See the `RandomWalk` documentation for details.
    integrator : callable, optional
        The integrator for the Hamilton equations. Defaults to `leapfrog`; for expected arguments
        and return value, see there -- its non-optional arguments are exactly how an integrator is
        invoked.
    """

    State = HamiltonState
    """The state class is `HamiltonState`."""

    def __init__(self, logpdf, stepsize, state=None, stepsize_rule=None, integrator=leapfrog):
        super().__init__(logpdf, stepsize, state=state, stepsize_rule=stepsize_rule)
        self.integrator = integrator
        """The integrator."""

    def _propose(self, state):
        return self.integrator(
            logpdf=self.logpdf,
            state=state.update(
                momenta=self.logpdf.domain.randn()
            ),
            stepsize=self.stepsize,
        )


class StateHistory:
    """Handles a list of states (actually a `collections.deque`) and an acceptance count. Intended
    to have the `add` method used as callback in `MetropolisHastings.run`, or in hand-written loops,
    to keep a number of final sampler states in memory.

    Parameters
    ----------
    maxlen : int
        Will be passes to `collections.deque` to determine the maximum number of remembered states.
        Older states will be removed.
    """

    def __init__(self, maxlen=None):
        self.states = deque(maxlen=int(maxlen))
        """The state history."""
        self.accepted = 0
        """The total number of accepted proposals."""
        self.rejected = 0
        """The total number of rejected proposals."""

    def add(self, state, accepted):
        """Add a new state if accepted, and update acceptance counts."""
        if accepted:
            self.accepted += 1
            self.states.append(state)
        else:
            self.rejected += 1

    @property
    def total(self):
        """The total number of proposals."""
        return self.rejected + self.accepted

    @property
    def acceptance_rate(self):
        """The acceptance ratio."""
        return self.accepted / self.total

    def samples(self):
        """Return an array of all remembered state positions."""
        # TODO returning the entire array is a bad idea once we implement histroy managers
        #      that store states on disk
        return np.array([s.pos for s in self.states])

    def logprobs(self):
        """Return an array of logprobs of the remembered states."""
        return np.array([s.logprob for s in self.states])

# TODO fix this
#
# from regpy.util.svd import randomized_svd
#
# class GaussianApproximation(object):
#     def __init__(self, pdf):
#         # find m_MAP by Maximum-Likelihood
#         # TODO: Insert one of the implemented solvers instead of scipy.optimize.minimize
#         # Is done in mcmc_second_variant.
#         # Insert approximated code to compute gamma_prior_half^{1/2}
#
#         self.pdf = pdf
#         self.stepsize = 'randomly chosen'
#         self.y_MAP = self.pdf.setting.op(self.pdf.initial_state.positions)
#         N = self.pdf.initial_state.positions.shape[0]
#         # define the prior-preconditioned Hessian
#         self.Hessian_prior = np.zeros((N, N))
#         self.gamma_prior_inv = np.zeros((N, N))
#         for i in range(0, N):
#             self.gamma_prior_inv[:, i] = self.pdf.prior.hessian(self.pdf.m_0, np.eye(N)[:, i])
#         D, S = np.linalg.eig(-np.linalg.inv(self.gamma_prior_inv))
#         D = D.real
#         S = S.real
#
#         self.gamma_prior_half = np.dot(S.transpose(), np.dot(np.diag(np.sqrt(D)), S))
#         #
#         for i in range(0, N):
#             self.Hessian_prior[:, i] = np.dot(
#                 self.gamma_prior_half,
#                 self.pdf.likelihood.hessian(self.pdf.m_0, np.dot(self.gamma_prior_half, np.eye(N)[:, i]))
#             )
#         # construct randomized SVD of Hessian_prior
#         self.L, self.V = randomized_svd(self, self.Hessian_prior)
#         self.L = -self.L
#         # define gamma_post
#         self.gamma_post = np.dot(
#             self.gamma_prior_half,
#             np.dot(
#                 self.V,
#                 np.dot(np.diag(1 / (self.L + 1)), np.dot(self.V.transpose(), self.gamma_prior_half))
#             )
#         )
#         self.gamma_post_half = np.dot(
#             self.gamma_prior_half,
#             np.dot(self.V, np.dot(np.diag(1 / np.sqrt(self.L + 1) - 1), self.V.transpose())) +
#             np.eye(self.gamma_prior_half.shape[0])
#         )
#
#     def random_samples(self):
#         R = np.random.normal(0, 1, self.gamma_post_half.shape[0])
#         m_post = self.pdf.initial_state.positions + np.dot(self.gamma_post_half, R)
#         return m_post
#
#     def next(self):
#         m_post = self.random_samples()
#         next_state = State()
#         next_state.positions = m_post
#         next_state.log_prob = np.exp(-np.dot(m_post, np.dot(self.gamma_post, m_post)))
#         self.state = next_state
#         return True
