"""Iterative solvers for inverse problems.
"""

from regpy.util import classlogger
from regpy.hilbert import HilbertSpace


class Solver:
    """Abstract base class for solvers. Solvers do not loop themselves, but are driven by
    repeatedly calling the `next` method. They expose the current iterate and value as attributes
    `x` and `y`, and can be iterated over, yielding the `(x, y)` tuple on every iteration (which
    may or may not be the same arrays as before, modified in-place).

    There are some convenience methods to run the solver with a `regpy.stoprules.StopRule`.

    Subclasses should override the method `_next(self)` to perform a single iteration. The main
    difference to `next` is that `_next` does not have a return value. If the solver
    converged, `converge` should be called, afterwards `_next` will never be called again. Most
    solvers will probably never converge on their own, but rely on the caller or a
    `regpy.stoprules.StopRule` for termination.
    """

    log = classlogger

    def __init__(self, callback=None):
        self.x = None
        """The current iterate."""
        self.y = None
        """The value at the current iterate. May be needed by stopping rules, but callers should
        handle the case when it is not available."""
        self.__converged = False
        self.callback = callback

    def converge(self):
        """Mark the solver as converged. This is intended to be used by child classes
        implementing the `_next` method.
        """
        self.__converged = True

    def next(self):
        """Perform a single iteration.

        Returns
        -------
        boolean
            False if the solver already converged and no step was performed.
            True otherwise.
        """
        if self.__converged:
            return False
        self._next()
        if self.callback != None:
            self.callback(self.x)
        return True

    def _next(self):
        """Perform a single iteration. This is an abstract method called from the public method
        `next`. Child classes should override it.

        The main difference to `next` is that `_next` does not have a return value. If the solver
        converged, `converge` should be called.
        """
        raise NotImplementedError

    def __iter__(self):
        """Return and iterator on the iterates of the solver.

        Yields
        ------
        tuple of arrays
            The (x, y) pair of the current iteration.
        """
        while self.next():
            yield self.x, self.y

    def until(self, stoprule=None):
        """Generator that runs the solver with the given stopping rule. This is convenience method
        that implements a simple generator loop running the solver until it either converges or the
        stopping rule triggers.

        Parameters
        ----------
        stoprule : regpy.stoprules.StopRule, optional
            The stopping rule to be used. If omitted, stopping will only be
            based on the return value of `next`.

        Yields
        ------
        tuple of arrays
            The (x, y) pair of the current iteration, or the solution chosen by
            the stopping rule.
        """
        for x, y in self:
            yield x, y
            if stoprule is not None and stoprule.stop(x, y):
                self.log.info('Stopping rule triggered.')
                # TODO document this behaviour
                yield x, y
                return
        self.log.info('Solver converged.')

    def run(self, stoprule=None):
        """Run the solver with the given stopping rule. This method simply runs the generator
        `regpy.solvers.Solver.until` and returns the final `(x, y)` pair.
        """
        for x, y in self.until(stoprule):
            pass
        return x, y


class HilbertSpaceSetting:
    """A Hilbert space *setting* for an inverse problem, used by e.g. Tikhonov-type solvers. A
    setting consists of

    - a forward operator,
    - a Hilbert space structure on its domain that measures the regularity of reconstructions, and
    - a Hilbert space structur on its codomain for the data misfit.

    This class  is mostly a container that keeps all of this data in one place and makes sure that
    the `regpy.hilbert.HilbertSpace.discr`s match the operator's domain and codomain.

    It also handles the case when the specified Hilbert space is actually an
    `regpy.hilbert.AbstractSpace` (or actually any callable) instead of a
    `regpy.hilbert.HilbertSpace`, calling it on the operator's domain or codomain to construct
    the concrete Hilbert space instances.

    Parameters
    ----------
    op : regpy.operators.Operator
        The forward operator.
    Hdomain, Hcodomain : regpy.hilbert.HilbertSpace or callable
        The Hilbert spaces or abstract spaces on the domain or codomain.
    """
    def __init__(self, op, Hdomain, Hcodomain):
        if not isinstance(Hdomain, HilbertSpace) and callable(Hdomain):
            Hdomain = Hdomain(op.domain)
        assert isinstance(Hdomain, HilbertSpace)
        assert Hdomain.discr == op.domain

        if not isinstance(Hcodomain, HilbertSpace) and callable(Hcodomain):
            Hcodomain = Hcodomain(op.codomain)
        assert isinstance(Hcodomain, HilbertSpace)
        assert Hcodomain.discr == op.codomain

        self.op = op
        """The operator."""
        self.Hdomain = Hdomain
        """The `regpy.hilbert.HilbertSpace` on the domain."""
        self.Hcodomain = Hcodomain
        """The `regpy.hilbert.HilbertSpace` on the codomain."""
