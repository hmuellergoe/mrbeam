'''
Created on Feb 22, 2012

@author: fmertens
'''

import numpy as np
import lightwise.nputils as nputils


def linear_chirp(x, i, a, b, c):
    return i * np.exp(1j * (a * x + b) * x + c)


def gaussien(n, nsigma):
    x = np.arange(-n / 2, n / 2)
    sigma = n / float(nsigma) / 2.
    return np.exp(- x ** 2 / (2. * sigma ** 2))


def gaussian(size, nsigma=None, width=None, center=None, center_offset=None):
    if nsigma is None and width is None:
        raise ValueError("You need to specify either nsigma or width")

    if width is not None:
        sigma = width / (2. * np.sqrt(2 * np.log(2)))
    else:
        sigma = size / nsigma / 2.

    if center is None:
        center = np.floor(size / 2.)
    if center_offset is not None:
        center += center_offset

    x = np.arange(size)
    p = [0, 1, center, sigma]

    return nputils.gaussian_fct(*p)(x)


def gaussien_w(n, width):
    return gaussien(n, n / float(width) * np.sqrt(2 * np.log(2)))


def dirac(n, t):
    x = np.zeros(2 * n + 1)
    x[n + t] = 1
    return x


def square(n, wide):
    x = np.zeros(n)
    b = (n - wide) / 2.
    x[b:-b] = 1
    return x


def lorentzian(size, gamma=None, center=None, center_offset=None):
    if center is None:
        center = np.floor(size / 2.)
    if center_offset is not None:
        center += center_offset

    x = np.arange(size)
    p = [0, 1, center, gamma]

    return nputils.lorentzian_1d(*p)(x)


if __name__ == '__main__':
    class Array:
        def __init__(self, data):
            self.data = data

        def copy(self):
            return Array(self.data.copy())

        def set_data(self, data):
            self.data = data

    import wds
    import matplotlib.pyplot as plot

    # l = lorentzian(500, gamma=10., center=200)
    i = gaussian(100, width=5, center=50)
    # i += gaussian(100, width=2, center=65)
    # i += nputils.gaussian_noise(i.shape, 0, 0.25)

    f = wds.FeaturesFinder(Array(i), 0.5)
    for segments in f.execute():
        print(segments)

    print("DD:", f.direct_detection())

    for v in i:
        print(v)

    plot.plot(i)
    plot.grid()
    plot.show()
