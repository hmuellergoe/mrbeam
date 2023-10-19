import numpy as np


def test_adjoint(op, tolerance=1e-10):
    """Numerically test validity of :meth:`adjoint` method.

    Checks if ::

        inner(y, op(x)) == inner(op.adjoint(x), y)

    in :math:`L^2` up to some tolerance for random choices of `x` and `y`.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator.
    tolerance : float, optional
        The maximum allowed difference between the inner products. Defaults to
        1e-10.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    x = op.domain.randn()
    fx = op(x)
    y = op.codomain.randn()
    fty = op.adjoint(y)
    err = np.real(np.vdot(y, fx) - np.vdot(fty, x))
    assert np.abs(err) < tolerance, 'err = {}'.format(err)


def test_derivative(op, steps=[10**k for k in range(-1, -8, -1)]):
    x = op.domain.rand()
    y, deriv = op.linearize(x)
    h = op.domain.rand()
    normh = np.linalg.norm(h)
    g = deriv(h)
    return [np.linalg.norm((op(x + step * h) - y) / step - g) / normh for step in steps]
