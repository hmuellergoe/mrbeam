from regpy.util import classlogger


class MissingValueError(Exception):
    pass


class StopRule:
    """Abstract base class for stopping rules.

    Attributes
    ----------
    x, y : arrays or `None`
        The chosen solution. Stopping rules may decide to yield a result
        different from the last iterate. Therefore, after :meth:`stop` has
        triggered, it should store the solution in this attribute. Before
        triggering, these attributes contain the iterates passed to
        :meth:`stop`.
    triggered: bool
        Whether the stopping rule decided to stop.
    """

    log = classlogger

    def __init__(self):
        self.x = None
        self.y = None
        self.triggered = False

    def stop(self, x, y=None):
        """Check whether to stop iterations.

        Parameters
        ----------
        x : array
            The current iterate.
        y : array, optional
            The operator value at the current iterate. Can be omitted if
            unavailable, but some implementations may need it.

        Returns
        -------
        bool
            `True` if iterations should be stopped.
        """
        if self.triggered:
            return True
        self.x = x
        self.y = y
        self.triggered = self._stop(x, y)
        return self.triggered

    def _stop(self, x, y=None):
        """Check whether to stop iterations.

        This is an abstract method. Child classes should override it.

        Parameters and return values are the same as for the public interface
        method :meth:`stop`.

        This method will not be called again after returning `True`.

        Child classes that need `y` should raise :class:`MissingValueError` if
        called with `y=None`.
        """
        raise NotImplementedError

    def __add__(self, other):
        return CombineRules([self, other])


class CombineRules(StopRule):
    """Combine several stopping rules into one.

    The resulting rule triggers when any of the given rules triggers and
    delegates selecting the solution to the active rule.

    Parameters
    ----------
    rules : list of :class:`StopRule`
        The rules to be combined.
    op : :class:`~regpy.operators.Operator`, optional
        If any rule needs the operator value and none is given to :meth:`stop`,
        the operator is used to compute it.

    Attributes
    ----------
    rules : list of :class:`StopRule`
        The combined rules.
    op : :class:`~regpy.operators.Operator` or `None`
        The forward operator.
    active_rule : :class:`StopRule` or `None`
        The rule that triggered.
    """

    def __init__(self, rules, op=None):
        super().__init__()
        self.rules = []
        for rule in rules:
            if type(rule) is type(self) and rule.op is self.op:
                self.rules.extend(rule.rules)
            else:
                self.rules.append(rule)
        self.op = op
        self.active_rule = None

    def __repr__(self):
        return 'CombineRules({})'.format(self.rules)

    def _stop(self, x, y=None):
        for rule in self.rules:
            try:
                triggered = rule.stop(x, y)
            except MissingValueError:
                if self.op is None or y is not None:
                    raise
                y = self.op(x)
                triggered = rule.stop(x, y)
            if triggered:
                self.log.info('Rule {} triggered.'.format(rule))
                self.active_rule = rule
                self.x = rule.x
                self.y = rule.y
                return True
        return False


class CountIterations(StopRule):
    """Stopping rule based on number of iterations.

    Each call to :attr:`stop` increments the iteration count by 1.

    Parameters
    ----------
    max_iterations : int
        The number of iterations after which to stop.
    """

    def __init__(self, max_iterations):
        super().__init__()
        self.max_iterations = max_iterations
        self.iteration = 0

    def __repr__(self):
        return 'CountIterations(max_iterations={})'.format(self.max_iterations)

    def _stop(self, x, y=None):
        self.iteration += 1
        self.log.info(
            'iteration = {} / {}'
            .format(self.iteration, self.max_iterations))
        return self.iteration >= self.max_iterations


class Discrepancy(StopRule):
    """Morozov's discrepancy principle.

    Stops at the first iterate at which the residual is smaller than a
    pre-determined multiple of the noise level::

        ||y - data|| < tau * noiselevel

    Parameters
    ----------
    norm : callable
        The norm with respect to which the discrepancy should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    data : array
        The right hand side (noisy data).
    noiselevel : float
        An estimate of the distance from the noisy data to the exact data.
    tau : float, optional
        The multiplier; must be larger than 1. Defaults to 2.
    """

    def __init__(self, norm, data, noiselevel, tau=2):
        super().__init__()
        self.norm = norm
        self.data = data
        self.noiselevel = noiselevel
        self.tau = tau

    def __repr__(self):
        return 'Discrepancy(noiselevel={}, tau={})'.format(
            self.noiselevel, self.tau)

    def _stop(self, x, y=None):
        if y is None:
            raise MissingValueError
        residual = self.data - y
        discrepancy = self.norm(residual)
        rel = discrepancy / self.noiselevel
        self.log.info('relative discrepancy = {}, tolerance = {}'.format(rel, self.tau))
        return rel < self.tau
    
class DiscrepancyFunctional(StopRule):
    def __init__(self, functional, tau):
        super().__init__()
        self.functional = functional
        self.tau = tau

    def __repr__(self):
        return 'DiscrepancyFunctional(tau={})'.format(
            self.tau)

    def _stop(self, x, y=None):
        res = self.functional(x)
        self.log.info('Functional --> {}'.format( res ))
        return res < self.tau


class RelativeChangeData(StopRule):
    """Stops if the relative change in the residual becomes small

    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined cutoff::

        ||y_k-y_{k+1}|| < delta

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    data : np array
        The data array
    """

    def __init__(self, norm, data, cutoff):
        super().__init__()
        self.norm = norm
        self.cutoff = cutoff
        self.data_old = data

    def __repr__(self):
        return 'RelativeChangeData(cutoff={})'.format(
            self.cutoff)

    def _stop(self, x, y=None):
        if y is None:
            raise MissingValueError
        change = self.norm(y - self.data_old)
        self.data_old = y
        self.log.info('RelativeChangeData = {}, cutoff = {}'.format(
            change, self.cutoff))
        return change < self.cutoff



class RelativeChangeSol(StopRule):
    """Stops if the relative change in the solution space becomes small

    Stops at the first iterate at which the difference between the old estimate
    and the new estimate is smaller than a pre-determined cutoff::

        ||y_k-y_{k+1}|| < delta

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    init : np array
        initial guess
    """

    def __init__(self, norm, init, cutoff):
        super().__init__()
        self.norm = norm
        self.cutoff = cutoff
        self.sol_old = init

    def __repr__(self):
        return 'RelativeChangeSol(cutoff={})'.format(
            self.cutoff)

    def _stop(self, x, y=None):
        change = self.norm(x - self.sol_old)
        self.sol_old = x
        self.log.info('RelativeChangeSol = {}, cutoff = {}'.format(
            change, self.cutoff))
        return change < self.cutoff


class Monotonicity(StopRule):
    """Stops if the residual is growing again.

        Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~regpy.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    data : np array
        The data array
    init_data : np array
        initial guess in data space
    """

    def __init__(self, norm, data, init_data):
        self.norm = norm
        self.data = data
        self.residual = self.norm(self.data - init_data)

    def __repr__(self):
        return 'Monotonicty'

    def _stop(self, x, y=None):
        if y is None:
            raise MissingValueError
        residual = self.norm(self.data - y)
        change = self.residual - residual
        self.residual = residual
        self.log.info('Monotonicity = {}'.format(
            change))
        return change < 0

class Display(StopRule):
    def __init__(self, functional, string):
        super().__init__()
        self.functional = functional
        self.string = string
    
    def __repr__(self):
        return 'Display'

    def _stop(self, x, y=None):
        self.log.info(self.string + '--> {}'.format( self.functional(x) ))
        return False