'''
Created on Feb 6, 2012

@author: fmertens
'''

import re

import numpy as np
import lightwise.wavelets_coefficients as wc

from lightwise import nputils


class WaveletFamilyBase(object):

    def __init__(self, name, orders):
        self.name = name
        self.orders = orders

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def get_name(self):
        return self.name

    def get_orders(self):
        keys = self.orders.keys()
        #nputils.sort_nicely(keys)
        return keys

    def is_from(self, wavelet):
        return wavelet.get_name() in self.get_orders()

    def get_wavelet(self, order):
        pass


class DiscreteWaveletFamilyBase(WaveletFamilyBase):

    def __init__(self, name, orders):
        WaveletFamilyBase.__init__(self, name, orders)

    def get_wavelet(self, order):
        return DiscreteWaveletBase(order, self.orders[order], self)


class DaubechiesWaveletFamily(DiscreteWaveletFamilyBase):

    def __init__(self):
        DiscreteWaveletFamilyBase.__init__(self, "Daubechies", wc.daubechies)


class SymletWaveletFamily(DiscreteWaveletFamilyBase):

    def __init__(self):
        DiscreteWaveletFamilyBase.__init__(self, "Symlets", wc.symlets)


class CoifletWaveletFamily(DiscreteWaveletFamilyBase):

    def __init__(self):
        DiscreteWaveletFamilyBase.__init__(self, "Coiflets", wc.coiflets)


class TriangeWaveletFamily(DiscreteWaveletFamilyBase):

    def __init__(self):
        DiscreteWaveletFamilyBase.__init__(self, 'Triange', wc.triange)


class BSplineWaveletFamily(DiscreteWaveletFamilyBase):

    def __init__(self):
        DiscreteWaveletFamilyBase.__init__(self, 'B Spline', wc.bspline)


class WaveletBase(object):

    def __init__(self, name, family):
        self.name = name
        self.family = family

    def __str__(self):
        return self.name

    def get_name(self):
        return self.name

    def get_family(self):
        return self.family


class DiscreteWaveletBase(WaveletBase):

    def __init__(self, name, hkd, family):
        WaveletBase.__init__(self, name, family)
        self.hk = hkd
        self.gk = nputils.qmf(hkd)

        # for cache
        self.wavelet_fct = None
        self.wavelet_fct_level = 0

    def get_max_level(self, data):
        # print data.shape, np.log2(np.array(data.shape).min()), len(self.get_dec_hk())
        return int(np.log2(np.array(data.shape).min() / len(self.get_dec_hk())))

    def get_rec_hk(self):
        return self.hk

    def get_dec_hk(self):
        return self.hk[::-1]

    def get_rec_gk(self):
        return self.gk

    def get_dec_gk(self):
        return self.gk[::-1]

    def get_wavelet_fct(self, level):
        # try to get it from cache
        if self.wavelet_fct_level == level and self.wavelet_fct is not None:
            return self.wavelet_fct

        N = len(self.hk) - 1

        #tabbed coef
        hk = self.get_rec_hk()
        gk = self.get_rec_gk()
        tab_hk = np.r_[0, hk]
        tab_gk = np.r_[0, gk]

        # building the matrix
        m = [0, 0]
        p = [0, 0]
        nn, kk = np.ogrid[1:N + 1, 1:N + 1]

        index_m0 = nputils.clipreplace(2 * nn - kk, 1, N + 1, 0)
        index_m1 = nputils.clipreplace(2 * nn - kk + 1, 1, N + 1, 0)
        alpha = 2 / np.sum(hk)

        m[0] = alpha * np.take(tab_hk, index_m0)
        m[1] = alpha * np.take(tab_hk, index_m1)

        p[0] = alpha * np.take(tab_gk, index_m0)
        p[1] = alpha * np.take(tab_gk, index_m1)

        x = np.arange(-N/2., N/2., pow(2, -level))
        phi = np.zeros_like(x)
        psi = np.zeros_like(x)
        v = dict()

        # compute the starting point
        al, av = np.linalg.eig(m[0])
        v_0 = av[:, np.argmin(abs(al - 1))]
        v[0] = v_0 / v_0.sum()

        pointsDone = np.empty(0)

        for j in range(1, int(level + 1)):
            points = np.setdiff1d(np.arange(0, 1, pow(2, -j)), pointsDone)
            pointsDone = np.r_[points, pointsDone]
            for point in points:
                im = int(np.floor((point + 0.5)))
                iv = (2 * point) % 1
                v[point] = np.dot(m[im], v[iv])
                phi[point * pow(2, level)::pow(2, level)] = np.real(v[point])
                psi[point * pow(2, level)::pow(2, level)] = np.real(np.dot(p[im], v[iv]))

        # cache result
        self.wavelet_fct = (x, phi, psi)
        self.wavelet_fct_level = level

        return (x, phi, psi)

    def get_tf_wavelet_fct(self, level, fmin=0, fmax=0, n=10000):
        x, phi, psi = self.get_wavelet_fct(level)

        f = np.linspace(-1 / (2 * pow(2, -level)),
                        1 / (2 * pow(2, -level)), n)

        tf_phi = np.fft.fftshift((np.fft.fft(phi, n)))
        tf_psi = np.fft.fftshift((np.fft.fft(psi, n)))

        if fmax > fmin:
            good = (f >= fmin) & (f <= fmax)
            f = f[good]
            tf_phi = tf_phi[good]
            tf_psi = tf_psi[good]

        tf_phi = tf_phi / max(tf_phi)
        tf_psi = tf_psi / max(tf_psi)

        return f, tf_phi, tf_psi

    def get_2d_wavelet_fc(self, level):
        x, phi, psi = self.get_wavelet_fct(level)

        psi1 = np.outer(phi, psi)
        psi2 = np.outer(psi, phi)
        psi3 = np.outer(psi, psi)
        phi = np.outer(phi, phi)

        return (x, phi, psi1, psi2, psi3)

    def get_2d_tf_wavelet_fct(self, level, fmin=0, fmax=0, n=10000):
        f, tf_phi, tf_psi = self.get_tf_wavelet_fct(level, fmin, fmax, n)

        tf_psi1 = np.outer(tf_phi, tf_psi)
        tf_psi2 = np.outer(tf_psi, tf_phi)
        tf_psi3 = np.outer(tf_psi, tf_psi)
        tf_phi = np.outer(tf_phi, tf_phi)

        return (f, tf_phi, tf_psi1, tf_psi2, tf_psi3)


def get_all_wavelet_families():
    return [DaubechiesWaveletFamily(), SymletWaveletFamily(),
            CoifletWaveletFamily(), TriangeWaveletFamily(),
            BSplineWaveletFamily()]


def get_wavelet(name):
    for family in get_all_wavelet_families():
        for order in family.get_orders():
            if order == name:
                return family.get_wavelet(order)
    return None
