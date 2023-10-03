import pytest
import numpy as np

from regpy import operators, util, discrs
from regpy.util import tests


def do_linear_test(op):
    for _ in range(10):
        tests.test_adjoint(op)


def do_nonlinear_test(op):
    for _ in range(10):
        large, small = tests.test_derivative(op, steps=[1e-1, 1e-8])
        assert small / large < 1e-5
    for _ in range(10):
        x = op.domain.rand()
        _, deriv = op.linearize(x)
        do_linear_test(deriv)


def test_linear_volterra():
    do_linear_test(
        operators.Volterra(
            domain=discrs.UniformGrid(np.linspace(0, 2 * np.pi, 200))))


def test_nonlinear_volterra():
    do_nonlinear_test(
        operators.Volterra(
            domain=discrs.UniformGrid(np.linspace(0, 2 * np.pi, 200)),
            exponent=3))


def test_mediumscattering():
    do_nonlinear_test(
        operators.MediumScattering(
            gridshape=(65, 65),
            radius=1,
            wave_number=1,
            inc_directions=util.linspace_circle(16),
            meas_directions=util.linspace_circle(16),
            amplitude=False))
