'''
Description: Set of class and function that complement numpy and scipy functions
(and a bit more...)

Created on Feb 6, 2012

@author: fmertens
'''

import os
import re
import math
import pickle
import calendar
import datetime
import itertools
import collections
import configparser

#import pymorph
import numpy as np
from scipy import optimize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import convolve1d as scipy_convolve1d
from scipy.optimize import leastsq, curve_fit
from scipy.ndimage.morphology import grey_dilation
from scipy.ndimage.measurements import center_of_mass

# inline import:
# heavy and rarely used: from scipy import signal
# heavy and rarely used: from scipy.cluster.hierarchy import fcluster

CONV_BOUNDARY_MAP = {"zero": "fill",
                     "symm": "symm",
                     "wrap": "wrap",
                     "border": "border"}

CONV_BOUNDARY_MAP2 = {"zero": "constant",
                      "symm": "reflect",
                      "wrap": "wrap",
                      "border": "border"}

K_CONV = 1.2E-8

K_FFT = 6E-9

si_prefix = {-3: "nano",
             -2: "micro",
             -1: "m",
             0: "",
             1: "k",
             2: "M",
             3: "G",
             4: "T"}

RANDOM_GENERATOR = np.random

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

np.seterr(divide='ignore', invalid='ignore')


class LimitedSizeDict(collections.OrderedDict):
    ''' From: http://stackoverflow.com/questions/2437617/limiting-the-size-of-a-python-dictionary '''

    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        collections.OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class Cache(LimitedSizeDict):

    def __init__(self, max_items):
        LimitedSizeDict.__init__(self, size_limit=max_items)


CACHE_SECROSS_FOOTPRINT = Cache(10)


def get_secross_footprint(size):
    #if size not in CACHE_SECROSS_FOOTPRINT:
    #    CACHE_SECROSS_FOOTPRINT[size] = pymorph.secross(r=int(size))
    #return CACHE_SECROSS_FOOTPRINT[size]
    raise NotImplementedError

def set_random_seed(seed):
    global RANDOM_GENERATOR
    RANDOM_GENERATOR = np.random.RandomState(seed)


def get_random(seed=None):
    if seed is None:
        return RANDOM_GENERATOR
    return np.random.RandomState(seed)


def time_conv2(shape1, shape2):
    return K_CONV * np.prod(shape1) * np.prod(shape2)


def time_fft2(shape):
    return K_FFT * np.prod(shape) * np.log(np.prod(shape))


def shift2d(array, delta):
    result = np.zeros_like(array)
    array_slice = []
    result_slice = []

    for dim in range(array.ndim):
        if int(delta[dim]) == 0:
            result_slice.append(slice(None, None))
            array_slice.append(slice(None, None))
        elif delta[dim] > 0:
            result_slice.append(slice(int(np.round(delta[dim])), None))
            array_slice.append(slice(None, -int(np.round(delta[dim]))))
        else:
            result_slice.append(slice(None, int(np.round(delta[dim]))))
            array_slice.append(slice(-int(np.round(delta[dim])), None))

    result[result_slice] = array[array_slice]
    return result


def coord_max(array, fit_gaussian=False, fit_gaussian_n=3):
    m = np.unravel_index(np.nanargmax(array), array.shape)

    if fit_gaussian:
        m = fit_gaussian_on_max(array, coord=tuple(m), n=fit_gaussian_n)
    return m


def coord_min(array, fit_gaussian=False, fit_gaussian_n=3):
    m = np.unravel_index(np.nanargmin(array), array.shape)

    if fit_gaussian:
        m = fit_gaussian_on_max(- array, coord=tuple(m), n=fit_gaussian_n)
    return m


def local_max(array, p, tol, fit_gaussian=False, fit_gaussian_n=3):
    clip = lambda p: (clamp(p[0], 0, array.shape[0]), clamp(p[1], 0, array.shape[1]))
    index = clip(np.array(p) - tol) + clip(np.array(p) + tol)
    tol_array = array[index2slice(index)]
    cmax = coord_max(tol_array, fit_gaussian=fit_gaussian, fit_gaussian_n=fit_gaussian_n) 
    cmax += np.array([index[0], index[1]])
    return tol_array.max(), cmax


def in_range(a, range):
    return a >= range[0] and a <= range[1]


def quantities_to_array(list, unit=None):
    unit = list[0].unit
    return np.array([k.to(unit).value for k in list]) * unit


def get_line_between_points(array, p1, p2, order=1):
    y0, x0 = p1
    y1, x1 = p2
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    z = map_coordinates(array, np.vstack((y, x)), order=order)

    return x, y, z


# def get_points_around(img, point, n=1, direction=None):
#     points = []
#     for (i, j) in get_offsets_around(n=n, direction=direction):
#         p = (point[0] + i, point[1] + j)
#         if check_index(img, p[0], p[1]):
#             points.append(p)
#     return points


def get_ellipse_radius(a, b, theta):
    ''' See http://math.stackexchange.com/questions/432902/how-to-get-the-radius-of-an-ellipse-at-a-specific-angle-by-knowing-its-semi-majo '''
    
    return a * b / np.sqrt(a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2)


def get_ellipse_minor_major(r, theta, a_b_ratio):
    b = r * np.sqrt((np.cos(theta) / a_b_ratio) ** 2 + np.sin(theta) ** 2)
    a = b * a_b_ratio
    return a, b


def create_ellipse(r, xc, alpha, n=100, angle_range=(0,2*np.pi)):
    """
    Create points on an ellipse with uniform angle step

    From https://code.google.com/p/fit-ellipse/source/browse/trunk/fit_ellipse.py

    Author: Alexis Mignon; Licence: GPL v3
    
    Parameters
    ----------
    r: tuple
        (rx, ry): major an minor radii of the ellipse. Radii are supposed to
        be given in descending order. No check will be done.
    xc : tuple
        x and y coordinates of the center of the ellipse
    alpha : float
        angle between the x axis and the major axis of the ellipse
    n : int, optional
        The number of points to create
    angle_range : tuple (a0, a1)
        angles between which points are created.
        
    Returns
    -------
        (n * 2) array of points """
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    
    a0,a1 = angle_range
    angles = np.linspace(a0,a1,n)
    X = np.vstack([ np.cos(angles) * r[0], np.sin(angles) * r[1]]).T
    return np.dot(X,R.T) + xc

# def get_offsets_around(n=1, direction=None):
#     for i in np.arange(-n, n + 1):
#         for j in np.arange(-n, n + 1):
#             if (i == 0 and j == 0):
#                 continue
#             if direction is not None:
#                 pdir = (i / np.abs(i), j / np.abs(j))
#                 diff = np.abs(pdir[0] - direction[0]) + np.abs(pdir[1] - direction[1])
#                 if diff > 1:
#                     continue
#             yield (i, j)


# def get_robinson3_operator(direction=None):
#     b = np.array([1, 0, -1])
#     east = np.array([b, b, b])
#     northeast = np.array([[0, -1, -1], b, [1, 1, 0]])
#     north = - east.T
#     northwest = np.array([[-1, -1, 0], -b, [0, 1, 1]])
#     west = - east
#     southwest = - northeast
#     south = - north
#     southest = - northwest
#     d = {(1, 0): east,
#          (1, 1): northeast,
#          (0, 1): north,
#          (-1, 1): northwest,
#          (-1, 0): west,
#          (-1, -1): southwest,
#          (0, -1): south,
#          (1, -1): southest}

#     if direction is None:
#         return d
#     return d[direction]


# def get_prewitt_compass_operator(direction=None):
#     d = dict()
#     for direction, op in get_robinson3_operator().items():
#         op[1, 1] = -2
#         op[op == 0] = 1
#         d[direction] = op
    
#     if direction is None:
#         return d
#     return d[direction]


# def get_integral_compass_operator(direction=None):
#     d = dict()
#     for direction, op in get_robinson3_operator().items():
#         op[op == -1] = 1
#         d[direction] = op
    
#     if direction is None:
#         return d
#     return d[direction]


# def get_values(array, coords):
#     res = []
#     for coord in coords:
#         res.append(array[tuple(coord)])
#     return res


# def repeat(v, n):
#     return [v] * n


def expend_slice(slice_obj, shape, axis=None):
    if axis is None:
        return [slice_obj] * len(shape)
    s = [slice(None)] * len(shape)
    s[axis] = slice_obj
    return s


def get_index(array, slice_obj, axis=None):
    '''
    Return a sliced array.

    @param array:
    @param slice_obj:
    @param axis:

    @UT: OK
    '''
    return np.asarray(array)[expend_slice(slice_obj, array.shape, axis=axis)]


def safe_strip(s):
    return s.strip() if s is not None else ""


def check_index(array, *index):
    for i, j in enumerate(index):
        if j < 0 or j >= array.shape[i]:
            return False
    return True


def shape_eq(array1, array2):
    if array1.ndim != array2.ndim:
        return False
    for s1, s2 in zip(array1.shape, array2.shape):
        if s1 != s2:
            return False
    return True


def random_walk(maxn=10000, delta=1):
    return np.cumsum(get_random().uniform(-delta / 2., delta / 2., (maxn, 1)))


def random_walk_fct(maxn=10000, delta=1):
    walk = random_walk(maxn, delta)

    return lambda i: walk[i]


def number_day_in_year(year):
    if calendar.isleap(year):
        return 366
    return 365


def datetime_to_epoch(date):
    if isinstance(date, datetime.date):
        day_of_the_year = date.timetuple().tm_yday
        return "%.5f" % (date.year + (day_of_the_year - 1) / 365.25)
    return date


def epoch_to_datetime(epoch):
    i, d = divmod(float(epoch), 1)
    if i < 1000 or i > 3000:
        return float(epoch)
    day = min(round((d * 365.25) + 1), 366)
    return datetime.datetime.strptime("%i %i" % (i, day), "%Y %j")


def datetime_to_mjd(datetime):
    from astropy.time import Time

    return Time(datetime, scale='utc').mjd


def mjd_to_datetime(mjd):
    return datetime.datetime(1858, 11, 17) + datetime.timedelta(mjd)


def guess_date(date_str, formats):
    for format in formats:
        try:
            return datetime.datetime.strptime(date_str, format)
        except ValueError:
            pass
    return None


def date_filter(start_date=None, end_date=None, filter_dates=None):

    def filter(date):
        if start_date is not None and date < start_date:
            return False
        if end_date is not None and date > end_date:
            return False
        if filter_dates is not None and date in filter_dates:
            return False
        return True

    return filter


def parsec_to_meter(parsec):
    return parsec * 3.085678 * 1e16


def meter_to_parsec(meter):
    return meter / 3.085678 * 1e16


def coord_rtheta_to_xy(r, theta):
    ''' Theta in radian, (East of North). Can be [- pi, pi ] or [ 0, 2 * pi ] '''
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.array([x, y])


def coord_xy_to_rtheta(x, y):
    ''' Return theta in radian, angle between [x, y] and [0, 1] (East of North).
        theta between [- pi, pi ] '''
    theta = np.arctan2(x, y)
    r = l2norm(np.array([x, y]), axis=0)
    return np.array([r, theta])


def affine_transform(x0, x1, X0, X1):
    a = (X1 - X0) / float(x1 - x0)
    b = X0 - a * x0
    return lambda x: a * x + b, lambda X: (X - b) / a


def l2norm(array, axis=-1):
    return np.sum(np.abs(array) ** 2, axis=axis) ** (1. / 2)


def display_measure(value, unit):
    exponent = np.floor(math.log(value, 1000))
    exponent = max(exponent, min(si_prefix.keys()))
    exponent = min(exponent, max(si_prefix.keys()))
    quotient = float(value) / 1000 ** exponent
    num_decimals = 2
    prefix = si_prefix[exponent]
    format_string = '{0:.%sf} {1}{2}' % (num_decimals)
    return format_string.format(quotient, prefix, unit)


def _corr_convolve_fast(x, y, mode='same', method='auto'):
    M = np.array(x.shape) + np.array(y.shape) - 1
    M_fft = np.clip(nextpow2(M), 2, None)#, 2048)

    time_fft = 3 * time_fft2(M_fft)
    time_conv = time_conv2(x.shape, y.shape)

    # make sure we work with float64 array
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if (time_fft > time_conv or method == 'conv') and method != 'fft':
        corr = convolve(x, y, boundary='zero')
    else:
        X = np.fft.rfftn(x, M_fft)
        Y = np.fft.rfftn(y, M_fft)
        corr = np.fft.irfftn(X * Y)
        corr = resize(corr, M, 'left')

    index = []
    for dim in range(x.ndim):
        if mode == 'same':
            l = int( (y.shape[dim] - 1) / 2 )
            r = int( -((y.shape[dim] - 1) - l) )
            index.append(slice(l, r))
        elif mode == 'valid':
            index.append(slice(y.shape[dim] - 1, -(y.shape[dim] - 1)))
        else:
            index.append(slice(None, None))

    return corr[tuple(index)]


def xcorr_fast(x, y, mode='same', method='auto'):
    return _corr_convolve_fast(x, flip(y), mode=mode, method=method)


def fftconvolve(x, y, mode='same'):
    return _corr_convolve_fast(x, y, mode=mode, method='fft')


def phase_correlation(x, y):
    M = np.array(x.shape) + np.array(y.shape) - 1
    M_fft = np.clip(nextpow2(M), 2, None)#,2048)

    X = np.fft.fftn(x, M_fft)
    Y = np.fft.fftn(y, M_fft)

    R = (X * np.conj(Y)) / np.abs(X * np.conj(Y))

    csd = np.real(np.fft.ifftn(R))
    csd = resize(csd, M, 'left')

    return csd.real, np.array(csd.shape) - coord_max(csd) + 1


def local_sum(a, shape, mode="same"):
    '''See http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html '''
    # make sure we work with float64 array
    res = resize(a, np.array(a.shape) + 2 *
                 np.array(shape) - 1).astype(np.float64)

    for dim in range(a.ndim):
        cum = np.cumsum(res, dim)
        a = get_index(cum, slice(shape[dim], None), axis=dim)
        b = get_index(cum, slice(0, -shape[dim]), axis=dim)
        res = a - b

    index = []
    for dim in range(a.ndim):
        if mode == 'same':
            l = (shape[dim] - 1) / 2
            r = -((shape[dim] - 1) - l)
            index.append(slice(l, r))
        elif mode == 'valid':
            index.append(slice(shape[dim] - 1, -(shape[dim] - 1)))
        else:
            index.append(slice(None, None))

    return res[index]


def norm_xcorr2(x, y, mode="same", method='auto', replace_nan_to_zero=True, debug=False):
    '''
    Actually a zero mean normalized cross correlation
    '''
    # make sure we work with float64 array
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    tol = np.finfo(x.dtype).eps * 1000

    ny = local_sum(np.ones_like(x), y.shape, mode=mode)

    x_mean = local_sum(x, y.shape, mode=mode) / ny
    x2_mean = local_sum(x ** 2, y.shape, mode=mode) / ny

    y_mean = xcorr_fast(np.ones_like(x), y, mode=mode, method=method) / ny
    y2_mean = xcorr_fast(
        np.ones_like(x), y ** 2, mode=mode, method=method) / ny

    sigma_x = np.sqrt(x2_mean - x_mean ** 2)
    sigma_y = np.sqrt(y2_mean - y_mean ** 2)
    cov_xy = xcorr_fast(x, y, mode=mode) / ny - (x_mean * y_mean)

    denominator = sigma_x * sigma_y

    nxcorr = np.where(
        denominator < tol + np.isnan(denominator), 0, cov_xy / denominator)

    if replace_nan_to_zero:
        nxcorr = np.nan_to_num(nxcorr)

    if debug is True:
        return nxcorr, cov_xy, denominator, xcorr_fast(np.ones_like(x), y, mode=mode, method=method)

    return nxcorr


def zero_mean_xcorr2(x, y, mode="same", method='auto', replace_nan_to_zero=True, debug=False):
    '''
    zero mean cross correlation
    '''
    # make sure we work with float64 array
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    tol = np.finfo(x.dtype).eps * 1000

    ny = local_sum(np.ones_like(x), y.shape, mode=mode)

    x_mean = local_sum(x, y.shape, mode=mode) / ny
    y_mean = xcorr_fast(np.ones_like(x), y, mode=mode, method=method) / ny

    cov_xy = xcorr_fast(x, y, mode=mode) / ny - (x_mean * y_mean)

    return cov_xy


def fit_gaussian_on_max(array, n=3, coord=None):
    if coord is None:
        coord = coord_max(array, fit_gaussian=False)

    assert len(coord) == array.ndim

    slice_array = []
    for i, x in enumerate(coord):
        a, b = np.clip([x - n, x + n + 1], 0, array.shape[i])
        slice_array.append(slice(a, b))
    d = array[slice_array]

    c = coord_max(d, fit_gaussian=False)
    # p: height, center, sigma
    p = [0, d.max(), c, [1] * len(c)]
    res, _ = fitgaussian(d, p)
    res_c = np.array(res[2])
    new_coord = np.array(coord) + (res_c - c)
    return coord_clip(new_coord, array.shape)


def norm_xcorr_coef(x, y, delta=None):
    if delta is not None:
        y = shift2d(y, -delta)
    x = resize_like(x, y)

    cov_xy = ((x - x.mean()) * (y - y.mean())).sum() / float(x.size)

    return cov_xy / (x.std() * y.std())


def align_on_com(array1, array2):
    assert array1.ndim == array2.ndim

    com1 = center_of_mass(array1)
    com2 = center_of_mass(array2)
    shape = []
    pos1 = []
    pos2 = []
    for dim in range(array1.ndim):
        delta = com2[0] - com1[0]
        if delta >= 0:
            pos1.append(delta)
            pos2.append(0)
            shape.append(max(array2.shape[dim], array1.shape[dim] + delta))
        if delta < 0:
            pos2.append(-delta)
            pos1.append(0)
            shape.append(max(array1.shape[dim], array2.shape[dim] - delta))
    out1 = np.zeros(shape)
    fill_at(out1, pos1, array1)
    out2 = np.zeros(shape)    
    fill_at(out2, pos2, array2)

    return out1, out2


def weighted_norm_xcorr2(x, y, w, mode="same", method="auto", replace_nan_to_zero=True):
    wn = w / float(w.sum())

    weighted_mean_x = xcorr_fast(x, wn, mode)

    weighted_mean_y = (wn * y).sum()

    weighted_y = wn * (y - weighted_mean_y)

    weighted_cov_xy = xcorr_fast(x, weighted_y, mode) - weighted_mean_x * weighted_y.sum()
    weighted_cov_xx = xcorr_fast(x ** 2, wn, mode) - weighted_mean_x ** 2
    weighted_cov_yy = (((y - (y * wn).sum()) ** 2) * wn).sum()

    denominator = np.sqrt(weighted_cov_xx * weighted_cov_yy)

    tol = np.sqrt(np.finfo(denominator.dtype).eps)
    wnxcorr = np.where(denominator < tol, 0, weighted_cov_xy / denominator)

    if replace_nan_to_zero:
        np.nan_to_num(wnxcorr)

    return wnxcorr


def ssd_fast(x, y, mode="same", method="auto"):
    '''
    Fast sum of squared differences (SSD block matching) for n-dimensional arrays
    '''
    xcorr = xcorr_fast(x, y, mode=mode, method=method)

    local_sum_x2 = local_sum(x ** 2, y.shape, mode=mode)

    ysum2 = xcorr_fast(np.ones_like(x), y ** 2, mode=mode, method=method)

    ssd = local_sum_x2 + ysum2 - 2. * xcorr

    return ssd


def zero_ssd_fast(x, y, mode="same", method="auto"):
    '''
    Fast zero mean sum of squared differences (SSD block matching) for n-dimensional arrays
    '''
    ny = local_sum(np.ones_like(x), y.shape, mode=mode)
    x_mean = local_sum(x, y.shape, mode=mode) / ny
    y_mean = y.mean()

    x = x - x_mean
    y = y - y_mean

    xcorr = xcorr_fast(x, y, mode=mode, method=method)

    local_sum_x2 = local_sum(x ** 2, y.shape, mode=mode)

    ysum2 = xcorr_fast(np.ones_like(x), y ** 2, mode=mode, method=method)

    ssd = local_sum_x2 + ysum2 - 2. * xcorr

    return ssd


def weighted_ssd_fast(x, y, w, mode="same", method="auto"):
    '''
    Fast sum of squared differences (SSD block matching) for n-dimensional arrays
    '''
    # w = w / float(w.sum()) * w.size
    w = w / float(w.sum()) * (w > 0).sum()

    xcorr = xcorr_fast(x, y * w, mode=mode, method=method)

    local_sum_x2 = xcorr_fast(x ** 2, w, mode=mode, method=method)

    ysum2 = xcorr_fast(np.ones_like(x), y ** 2 * w, mode=mode, method=method)

    ssd = local_sum_x2 + ysum2 - 2. * xcorr

    return ssd


def norm_ssd_fast(x, y, mode="same", method="auto"):
    '''
    Fast sum of squared differences (SSD block matching) for n-dimensional arrays
    '''
    return 2 - 2 * norm_xcorr2(x, y, mode=mode, method=method)


def combinations_multiple_r(array, min_r=1, max_r=None):
    if max_r is None:
        max_r = len(array)
    for r in range(min_r, max_r + 1):
        for combi in itertools.combinations(array, r):
            yield combi


def uniq_subsets(s):
    u = set()
    for x in s:
        t = []
        for y in x:
            y = list(y)
            y.sort()
            t.append(tuple(y))
        t.sort()
        u.add(tuple(t))
    return u


def k_subset(s, k, filter=None):
    '''
    See http://codereview.stackexchange.com/questions/1526/python-finding-all-k-subset-partitions
    '''
    if k == len(s):
        return (tuple([(x,) for x in s]),)
    k_subs = []
    for i in range(len(s)):
        partials = k_subset(s[:i] + s[i + 1:], k, filter=filter)
        for partial in partials:
            for p in range(len(partial)):
                if filter is None or filter(partial[p] + (s[i],)):
                    k_subs.append(partial[:p] + (partial[p] + (s[i],),) + partial[p + 1:])
    return uniq_subsets(k_subs)


def all_k_subset(s, k, filter=None):
    for subset in combinations_multiple_r(s):
        for l in k_subset(subset, k, filter=filter):
            yield l


def lists_combinations(l1, l2, k=None, filter=None):
    '''
    See http://stackoverflow.com/questions/10887530/algorithm-to-retrieve-\
             every-possible-combination-of-sublists-of-a-two-lists
        http://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
    '''

    if k is not None:
        k = [k]
    else:
        k = range(1, min(len(l1), len(l2)) + 1)
    for k in k:
        for item in itertools.product(all_k_subset(l1, k, filter=filter), all_k_subset(l2, k, filter=filter)):
            for per in itertools.permutations(item[1]):
                yield item[0], per


def sublist(array, indices):
    return [array[int(i)] for i in indices]


def permutation_no_succesive(a):
    assert len(a) >= 3

    d2 = dict(zip(a[1:], a[:-1]))
    d2[a[0]] = -1
    while True:
        t = np.random.permutation(a)
        ok = True
        for t1, t2 in pairwise(t):
            if t1 == d2[t2]:
                ok = False
                continue
        if ok:
            break
    return t


def clamp(n, minn, maxn):
    ''' Clip value between min and max'''
    return max(min(maxn, n), minn)


def coord_clip(coord, shape):
    new_coord = []
    for c, cmax in zip(coord, shape):
        new_coord.append(clamp(c, 0, cmax - 1))
    return new_coord


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


def nwise(iterable, n):
    "s -> (s0,s1, s2), (s1,s2, s3), (s2, s3, s4), ..."
    ilist = itertools.tee(iterable, n)
    for i, it in enumerate(ilist):
        for i in range(i):
            next(it, None)
    return zip(*ilist)


def get_values_sorted_by_keys(dict):
    return [value for (key, value) in sorted(dict.items())]


def get_items_sorted_by_keys(dict):
    return [(key, value)  for (key, value) in sorted(dict.items())]


# def intersect_affine(a1, b1, a2, b2):
#     x = (b2 - b1) / (a1 - a2)
#     y = a1 * x + b1
#     return (x, y)


def affine_fct_from_angle_point(angle, point):
    x, y = point
    a = np.tan(angle)
    b = y - x * a
    return a, b


def vector_projection(vector, vector_component):
    vector = np.asarray(vector)
    vector_component = np.asarray(vector_component) / l2norm(vector_component, axis=0)
    orth_vector_component = np.array([-vector_component[1], vector_component[0]])
    v1 = np.atleast_1d(np.dot(vector.T, vector_component))
    if v1.ndim == 2:
        v1 = np.diag(v1)
    # v2 = np.linalg.norm(vector - v1 * vector_component)
    v2 = np.atleast_1d(np.dot(vector.T, orth_vector_component))
    if v2.ndim == 2:
        v2 = np.diag(v2)

    return np.array([v1, v2])


# def normalise(a, b):
#     '''
#     Considering that b = i * a + j, return (a - j) / i
#     '''
#     a = a - (a.mean() - b.mean())
#     a = a / a.std() * b.std()
#     return a - (a.mean() - b.mean())


# def scale_normalise(a, b):
#     '''
#     Considering that b = i * a + j, return (a - j) / i
#     '''
#     return a / a.std() * b.std()


# def fit_normalise(a, b):
#     '''
#     Considering that b = i * a + j, return (a - j) / i which minimize (b - (a - j) / i) ** 2
#     '''

#     def fct(p):
#         return (b - (p[0] * a + p[1])).ravel()

#     (i, j), n = optimize.leastsq(fct, [1, 0])

#     return (a * i) + j


# def scale_fit_normalise(a, b):
#     '''
#     Considering that b = i * a + j, return (a - j) / i which minimize (b - (a - j) / i) ** 2
#     '''

#     def fct(p):
#         return (b - (p[0] * a + p[1])).ravel()

#     (i, j), n = optimize.leastsq(fct, [1, 0])

#     return (a * i)


# def has_duplicates(array):
#     return len(set(array)) != len(array)


def count(array):
    ''' Return number of unique item in array in the form:
    ((item1, count1), (item2), count2), ...)'''
    count = np.bincount(array)
    items = np.nonzero(count)[0]
    return zip(items, count[items])


def uniq(array):
    return list(collections.OrderedDict.fromkeys(array))


def find_peaks(img, width, threashold, exclude_border=True, max_peaks=None, fit_gaussian=False, 
               fit_gaussian_n=3, exclude_border_dist=1):
    ''' Caveats: if peak is spread over 2 pixel with exact same intensity, then 2 peaks will be detected '''
    if img.ndim == 1:
        footprint = np.ones(width)
    else:
        footprint = get_secross_footprint(width)
    dilate = lambda signal: grey_dilation(signal, footprint=footprint)

    # those are the local maximums
    peak_mask = dilate(img) == img
    peaks = np.empty_like(img)
    peaks[:] = np.nan
    peaks[peak_mask] = img[peak_mask]

    # filter the peaks by threashold
    peaks[peaks < threashold] = np.nan

    # exclude border
    if exclude_border:
        peaks[-exclude_border_dist:, :] = np.nan
        peaks[:exclude_border_dist, :] = np.nan
        peaks[:, -exclude_border_dist:] = np.nan
        peaks[:, :exclude_border_dist] = np.nan

    peaks_coord = np.argwhere(~np.isnan(peaks))
    peaks_coord = sorted(peaks_coord, key=lambda peak: img[tuple(peak)], reverse=True)

    if max_peaks is not None:
        peaks_coord = peaks_coord[:max_peaks]

    if fit_gaussian:
        new_peaks_coord = []
        for peak in peaks_coord:
            new_peaks_coord.append(fit_gaussian_on_max(img, coord=tuple(peak), n=fit_gaussian_n))
        peaks_coord = new_peaks_coord

    # return peak_mask, list(peaks_coord)
    return list(peaks_coord)


def get_interface(labels, label):
    #PERF ISSUE: maybe crop the labels to make it smaller!
    labels = labels.copy()

    clabels0 = convolve(labels, [1, 1], axis=0, boundary='zero', mode='same')
    clabels1 = convolve(labels, [1, 1], axis=1, boundary='zero', mode='same')

    masked_labels = labels.copy()
    masked_labels[masked_labels != label] = 0

    masked_clabels0 = convolve(
        masked_labels, [1, -1], axis=0, boundary='zero', mode='same')
    masked_clabels1 = convolve(
        masked_labels, [1, -1], axis=1, boundary='zero', mode='same')

    results = dict()
    for cx, cy in np.array(np.nonzero(clabels0 * masked_clabels0)).T:
        l = clabels0[cx, cy] - label
        if l not in results:
            results[l] = []
        results[l].append([cx, cy])

    for cx, cy in np.array(np.nonzero(clabels1 * masked_clabels1)).T:
        l = clabels1[cx, cy] - label
        if l not in results:
            results[l] = []
        results[l].append([cx, cy])

    return results


def k_sigma_noise_estimation(data, k=3, iteration=3, beam=None):
    noise = gaussian_noise([int(10000 ** (1 / float(data.ndim)))] * data.ndim, 0, 1)
    if beam is not None:
        noise = beam.convolve(noise)
        detail = lambda a: a - smooth(a, max(beam.bmin, beam.bmaj) * 3, mode='same')
    else:
        detail = lambda a: a - smooth(a, 3, mode='same')
    sigma_filtered = detail(noise).std()

    d = detail(data)

    for i in range(iteration):
        d[np.abs(d) > k * d.std()] = 0

    return d.std() / sigma_filtered


# def preprocess_noise(noise, iteration=3, k=3):
#     for i in range(iteration):
#         noise[np.abs(noise) > k * noise.std()] = 0
#     return noise


# def match_template(image, template, tol=0.9):
#     corr = norm_xcorr2(image, template, mode='valid')

#     peaks = find_peaks(corr, 4, tol)

#     return corr, peaks


# @profileutils.line_profil
def crop_threshold_old(array, threashold=0, crop_mask=None, output_index=False):
    if crop_mask is None:
        crop_mask = array

    coords = np.array(np.where(crop_mask > threashold))

    if coords.size <= 1:
        x0 = y0 = 0
        x1, y1 = np.array(array.shape)
    else:
        x0, y0 = coords.min(axis=1)
        x1, y1 = coords.max(axis=1) + 1

    if output_index:
        return array[x0:x1, y0:y1], [x0, y0, x1, y1]
    return array[x0:x1, y0:y1]


def crop_threshold(array, threashold=0, crop_mask=None, output_index=False):
    if crop_mask is None:
        crop_mask = array

    coords = np.array(np.where(crop_mask > threashold))

    if coords.size <= 1:
        i1 = np.array(array.shape)
        i0 = np.zeros_like(i1)
    else:
        i0 = coords.min(axis=1)
        i1 = coords.max(axis=1) + 1

    index = i0.tolist() + i1.tolist()
    slices = index2slice(index)

    if output_index:
        return array[slices], index
    return array[slices]


def nextpow2(n):
    '''get the next power of 2 that's greater than n'''
    return 2 ** np.ceil(np.log2(n)).astype(int)


def angle(v1, v2):
    return np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def flip(a):
    '''Inverts an n-dimensional array along each of its axes'''
    ind = (slice(None, None, -1),) * a.ndim
    return a[ind]


def sort_index(list):
    return [i[0] for i in sorted(enumerate(list), key=lambda x:x[1])]


def tryint(s):
    ''' Try to return convert s to an int.

        From: http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    ''' Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]

        From: http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def sort_nicely(l):
    ''' Sort in place the given list in the way that humans expect.

        From: http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    l.sort(key=alphanum_key)


def distance_from_border(point, shape):
    return [point[0], shape[0] - point[0] - 1, point[1], shape[1] - point[1] - 1]


def save_object(object, filename, dir=DATA_DIR):
    ''' DEPRECATED '''
    if not os.path.exists(dir):
        os.makedirs(dir)

    path = os.path.abspath(os.path.join(dir, filename + '.data'))
    file = open(path, 'w')
    pickle.dump(object, file)
    file.close()
    print("Object saved to ", path)
    return path


def load_object(filename, dir=DATA_DIR):
    ''' DEPRECATED '''
    path = os.path.abspath(os.path.join(dir, filename + '.data'))
    print("Loading object from ", path)
    file = open(path, 'r')
    object = pickle.load(file)
    print("Object loaded.")
    return object


def clipreplace(array, mini, maxi, replace):
    '''
    Clip the array and replace the value points not in range to 'replace'.

    @param array:
    @param mini:
    @param maxi:
    @param replace:

    @UT: OK
    '''
    array[(array > maxi) + (array < mini)] = replace
    return array


def slice2index(slices):
    index = []
    for point in zip(*[[k.start, k.stop] for k in slices]):
        index.extend(point)
    return index


def index2slice(index):
    i = len(index) / 2
    return [slice(d0, d1) for d0, d1 in zip(index[:i], index[i:])]


def roundrobin(*iterables):
    '''
    roundrobin('ABC', 'D', 'EF') --> A D E B F C
    From: http://docs.python.org/2/library/itertools.html
    '''
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


def qmf(hkd):
    '''
    Return the Quadrature mirror filter hkd

    @param hkd:

    @UT: NO
    '''
    return [pow(-1, k) * hkd[len(hkd) - 1 - k] for k in range(0, len(hkd))]


def _check_axis(array, axis):
    if axis is not None and (axis > array.ndim or axis < 0):
        raise ValueError("Incorrect axis %s for array of dim %s" %
                         (axis, array.ndim))


def downsample(a, n, oddeven=0, axis=None):
    '''
    Downsample the array 'a' by a factor of 'n'.
    Return array is len floor(len(a) / n) if odd ceil(len(a) / n) if even

    @param a:
    @param n:
    @param axis: if axis is set, downsample only along this axis
    @param oddeven: if even, Y[k] = X[nk], if odd, Y[k] = X[nk+1]

    @UT: OK
    '''
    if n <= 0:
        raise ValueError("n should be > 0")
    a = np.asarray(a)
    _check_axis(a, axis)
    index = [np.s_[::]] * a.ndim
    for dim in range(a.ndim):
        if axis is None or dim == axis:
            index[dim] = np.s_[oddeven % 2::n]
    return a[index]


def upsample(a, n, oddeven=0, lastzero=False, axis=None):
    '''
    Upsample the array 'a' by a factor of 'n'.
    Return array array is 2 * len(a)

    @param a:
    @param n:
    @param end: if end is True, pad with zero at the end
    @param axis: if axis is set, upsample only along this axis
    @param oddeven: if even, Y[nK] = X[k], if odd Y[nk+1] = X[k]

    @UT: OK
    '''
    if n <= 0:
        raise ValueError("n should be > 0")
    a = np.asarray(a)
    _check_axis(a, axis)
    shape = list(a.shape)
    index = [np.s_[::]] * a.ndim
    for dim in range(a.ndim):
        if axis is None or dim == axis:
            shape[dim] = shape[dim] * int(n) + int(lastzero)
            index[dim] = np.s_[oddeven % 2::n]
    array = np.zeros(shape[:], dtype=a.dtype)
    array[index] = a
    return array


def atrou(a, n, axis=None):
    if n <= 0:
        raise ValueError("n should be > 0")
    a = np.asarray(a)
    _check_axis(a, axis)
    shape = list(a.shape)
    index = [np.s_[::]] * a.ndim
    for dim in range(a.ndim):
        if axis is None or dim == axis:
            shape[dim] = shape[dim] + (shape[dim] - 1) * int(n - 1)
            index[dim] = np.s_[::n]
    array = np.zeros(shape[:], dtype=a.dtype)
    array[tuple(index)] = a
    return array


# def per_extension(a, nleft, nright, axis=None):
#     '''
#    Extend the array 'a' by 'nleft' to the left and 'nright' to the right
#    by periodiazing the array.

#    @param a:
#    @param nleft:
#    @param nright:
#    @param axis: if axis is set, extend only along this axis

#    @UT: OK
#    '''
#     _check_axis(a, axis)
#     res = a
#     for dim in range(a.ndim):
#         concat = []
#         if axis is None or dim == axis:
#             if nleft != 0:
#                 concat.append(get_index(res, np.s_[-nleft::], dim))
#             concat.append(res)
#             if nright != 0:
#                 concat.append(get_index(res, np.s_[:nright:], dim))
#             res = np.concatenate(concat, axis=dim)
#     return res


# def symm_extension(a, nleft, nright, axis=None):
#     '''
#     Extend the array 'a' by 'nleft' to the left and 'nright' to the right
#     by symmetry.

#     @param a:
#     @param nleft:
#     @param nright:
#     @param axis: if axis is set, extend only along this axis

#     @UT: OK
#     '''
#     _check_axis(a, axis)
#     res = a
#     for dim in range(a.ndim):
#         concat = []
#         if axis is None or dim == axis:
#             if nleft != 0:
#                 concat.append(get_index(res, np.s_[nleft-1::-1], dim))
#             concat.append(res)
#             if nright != 0:
#                 concat.append(get_index(res, np.s_[-1:-1 - nright:-1], dim))
#             res = np.concatenate(concat, axis=dim)

#     return res


def fill_extension(a, nright, nleft, fillvalue=0, axis=None):
    '''
   Extend the array 'x' by 'nleft' to the left and 'nright' to the right
   by padding with zero.

   @param x:
   @param nleft:
   @param nright:
   @param axis: if axis is set, extend only along this axis

   @UT: OK
   '''
    _check_axis(a, axis)
    shape = list(a.shape)
    index = [np.s_[::]] * a.ndim
    for dim in range(a.ndim):
        if axis is None or dim == axis:
            shape[dim] = a.shape[dim] + nleft + nright
            if nleft == 0:
                index[dim] = np.s_[nright:]
            else:
                index[dim] = np.s_[nright:-nleft]
    res = np.ones(shape) * fillvalue
    res[tuple(index)] = a
    return res


def _convolve_1d_full(a, v, boundary='symm', mode='full'):
    from scipy.signal import convolve2d as scipy_convolve2d

    res = scipy_convolve2d([a], [v], mode=mode, boundary=CONV_BOUNDARY_MAP[boundary])
    return res[0]


def _convolve_1d(a, v, boundary='symm', mode='same', axis=0):
    assert mode in ['same', 'valid']
    assert v.ndim == 1
    res = scipy_convolve1d(a, v, mode=CONV_BOUNDARY_MAP2[boundary], axis=axis)
    
    if mode == 'valid':
        l = (len(v) - 1) / 2
        r = (len(v) - 1) - l
        index = [slice(None)] * res.ndim
        index[axis] = slice(l, -r)
        res = res[tuple(index)]
    return res


def convolve(a, v, boundary='symm', axis=None, mode='full', using_fft=True, using_scipy=True):
    '''
    Convolve signal a with kernel v.
    If a is 1D, perform a simple convolution
    If a is 2D, there is 3 possibility:
    - if v is 2D and axis=None, perform 2D conv using scipy convolve2d if using_fft is False, fftconvolve otherwise.
    - if v is 1D, perform 1D conv over each axis
    - if axis!=None, perform 1D convolution over this axis
    NOTE: axis!=None when v is 2D is not possible

    :param a:
    :param v:
    :param extension: extension funcion
    :param axis:
    :param mode:

    @UT: TODO:
    '''
    a = np.asarray(a)
    v = np.asarray(v)

    if boundary not in CONV_BOUNDARY_MAP.keys():
        raise ValueError("Wrong boundary")

    if v.ndim == 2 and axis is not None:
        raise ValueError("Convolution over an axis not possible if v is 2D")

    if v.ndim == 2 and a.ndim == 1:
        raise ValueError("2D v and 1D a not support.")

    if a.ndim == 2 and axis is None:
        if v.ndim == 1:
            c_r = convolve(a, v, boundary, axis=0, mode=mode)
            result = convolve(c_r, v, boundary, axis=1, mode=mode)
        elif v.ndim == 2:
            if using_fft:
                result = fftconvolve(a, v, mode=mode)
            else:
                from scipy.signal import convolve2d as scipy_convolve2d

                result = scipy_convolve2d(a, v, mode=mode, boundary=CONV_BOUNDARY_MAP[boundary])
        else:
            raise ValueError("Wrong dimension for v")
    else:
        if a.ndim == 1:
            if mode == 'full':
                result = _convolve_1d_full(a, v, boundary=boundary, mode=mode)
            else:
                result = _convolve_1d(a, v, mode=mode, boundary=boundary, axis=0)
        elif a.ndim == 2:
            if mode != 'full':
                result = _convolve_1d(a, v, mode=mode, boundary=boundary, axis=axis)
            else:
                # We don't know yet the dimension of the array, defer the
                # initialization
                result = None
                view = a
                # TODO: better way to handle that: transpose with axis and -1 flip
                # do same transposition at the end
                if axis == 0:
                    view = a.T
                for i in range(view.shape[0]):
                    line_conv = _convolve_1d_full(view[i], v, mode=mode, boundary=boundary)
                    if result is None:
                        result = np.zeros(
                            (view.shape[0], len(line_conv)), dtype=a.dtype)
                    result[i, :] = line_conv
                if axis == 0:
                    result = result.T
        else:
            raise ValueError("Wrong dimension for a: %s" % a.ndim)

    return result


def gaussian_noise(shape, mu, sigma):
    '''
    Create a gaussien noise

    :param shape:
    :param mu:
    :param sigma:

    @UT: OK
    '''
    n = np.array(shape).prod()
    return (mu + sigma * get_random().randn(n)).reshape(shape)


def fill_at(array, point, other, mode='replace', allow_exceed=True):
    '''
    Fill the array 'array' with the array 'other' at point 'point'.

    :param array:
    :param point:
    :param other:

    @UT: TODO:
    '''
    # print array.shape, point, other.shape, mode
    if array.ndim != other.ndim:
        raise ValueError("array and value should be of same dimension")
    if len(point) != array.ndim:
        raise ValueError("point should be of same dimension of array")
    index = []
    index_other = []
    for dim in range(array.ndim):
        aleft = point[dim]
        oleft = 0
        size = other.shape[dim]
        if point[dim] > array.shape[dim]:
            if not allow_exceed:
                raise ValueError("point outside of array")
            return
        if point[dim] + other.shape[dim] > array.shape[dim]:
            if not allow_exceed:
                raise ValueError("value does not fit in array")
            size = array.shape[dim] - point[dim]
        if point[dim] < 0:
            if not allow_exceed:
                raise ValueError("value does not fit in array")
            if point[dim] + size < 0:
                return
            aleft = 0
            oleft = - point[dim]
            size = size + point[dim]
        index.append(slice(aleft, aleft + size))
        index_other.append(slice(oleft, oleft + size))
    if mode == 'replace':
        array[tuple(index)] = other[tuple(index_other)]
    elif mode == 'add':
        array[tuple(index)] += other[tuple(index_other)]
    elif mode == 'sub':
        array[tuple(index)] -= other[tuple(index_other)]
    elif mode == 'mean':
        array[tuple(index)] += other[tuple(index_other)]
        array[tuple(index)] /= 2.
    elif mode == 'max':
        array[tuple(index)] = np.max([array[tuple(index)], other[tuple(index_other)]], axis=0)
    else:
        raise ValueError("mode '%s' does not exist" % mode)


def resize(array, shape, padding_mode='center', output_index=False):
    '''
    Resize the array 'array' to match shape 'shape'.

    For each dimension, if shape is smaller than array size, array is cut
    depending of padding_mode

    Padding mode: 'center', 'right', 'left'.

    If padding is impair, pad more to the left than to the right.

    :param array:
    :param shape:
    :param padding_mode:

    @UT: TODO:
    '''
    if array.ndim != len(shape):
        raise ValueError("array and like should have the same dimension")

    array = np.asarray(array)

    padding_slice = []
    index_slice = []

    for dim in range(array.ndim):
        diff = (array.shape[dim] - shape[dim])
        if padding_mode == 'right':
            nleft = int(abs(diff))
            nright = None
        elif padding_mode == 'left':
            nleft = None
            nright = int(-abs(diff))
        else:
            nleft = int(abs(np.floor(diff / 2.)))
            nright = int(-abs(np.ceil(diff / 2.)))
        if nleft == 0:
            nleft = None
        if nright == 0:
            nright = None
        if diff > 0:
            index_slice.append(slice(nleft, nright))
            padding_slice.append(slice(None))
        elif diff < 0:
            index_slice.append(slice(None))
            padding_slice.append(slice(nleft, nright))
        else:
            index_slice.append(slice(None))
            padding_slice.append(slice(None))

    # check if we need to reallocate (i.e., if new data are needed)
    if padding_slice == [slice(None)] * array.ndim:
        res = array[tuple(index_slice)]
    else:
        res = np.zeros(shape, dtype=array.dtype)

        res[padding_slice] = array[index_slice]
   
    if output_index is True:
        return res, slice2index(padding_slice), slice2index(index_slice)
    return res


def resize_like(array, like, padding_mode='center', output_index=False):
    '''
    Resize the array 'array' like the shape of array 'like'

    :param array:
    :param shape:
    :param padding_mode:

    @UT: TODO:
    '''
    return resize(array, like.shape, padding_mode=padding_mode, output_index=output_index)


def zoom(array, center, shape, pad=True, output_index=False, pad_value=0):
    '''
    Zoom the image around point 'center'.

    @param array:
    @param center:
    @param shape:

    @UT: TODO:
    '''

    if pad:
        array = fill_extension(array, max(shape), max(shape), fillvalue=pad_value)
        center = np.array(center) + max(shape)

    index_slice = []

    for dim in range(array.ndim):
        l = (shape[dim]) / 2
        r = shape[dim] - l
        left = max(0, center[dim] - l)
        right = max(0, min(center[dim] + r, array.shape[dim]))
        index_slice.append(slice(left, right))

    if output_index:
        return array[index_slice], slice2index(index_slice)

    return array[index_slice]


# def crop(array, index):
#     #TODO!!
#     shape = []
#     index_slice = []
#     array_slice = []

#     i = len(index) / 2

#     for dim, (d0, d1)  in zip(index[:i], index[i:]):
#         shape.append(d1 - d0)
#         array_slice.append(slice(max(0, -d0), min()))


def linear_fct(p1, p2):
    a = (p2[1] - p1[1]) / float(p2[0] - p1[0])
    b = p1[1] - a * p1[0]
    return lambda x: a * x + b


# def interpol_unstructured(points, values, method):
#     if method == 'nearest':
#         return interpolate.NearestNDInterpolator(points, values)
#     elif method == 'cubic':
#         return interpolate.CloughTocher2DInterpolator(points, values)
#     else:
#         return interpolate.LinearNDInterpolator(points, values)


# def rectopolar_coordinate_chunk(nx, ny):
#     x = range(nx)
#     for i in range(ny):
#         y = np.linspace(i, ny - 1 - i, nx)
#         yield np.vstack((x, y)).T
#     y = range(ny - 1, -1, -1)
#     for i in range(1, nx):
#         x = np.linspace(i, nx - 1 - i, ny)
#         yield np.vstack((x, y)).T


# def polar_coordinate(nx, ny):
#     origin = (nx // 2, ny // 2)
# Determine that the min and max r and theta coords will be...
#     x, y = projection.index_coords([nx, ny])
#     r, theta = projection.cart2polar(x, y)

# Make a regular (in polar space) grid based on the min and max r & theta
#     r_i = np.linspace(0, nx // 2, nx)
#     theta_i = np.linspace(theta.min(), theta.max(), ny)
#     theta_grid, r_grid = np.meshgrid(theta_i, r_i)

# Project the r and theta grid back into pixel coordinates
#     xi, yi = projection.polar2cart(r_grid, theta_grid)

# xi += origin[0]  # We need to shift the origin back to
# yi += origin[1]  # back to the lower-left corner...
#     xi, yi = xi.flatten(), yi.flatten()

# return np.vstack((xi, yi)).T  # (map_coordinates requires a 2xn array)


# def cartesian_coordinate_chunk(nx, ny, zero_center=False):
# #    if zero_center:
# #        x_left = -nx / 2
# #        x_right = nx / 2
# #        y_left = -ny / 2
# #        y_right = ny / 2
# #    else:
# #        x_left = 0
# #        x_right = nx - 1
# #        y_left = 0
# #        y_right = ny - 1
#     for i in range(0, nx):
#         yield np.array([[i] * ny, range(0, ny)]).T


# def cart_to_recto(img, method='linear'):
#     n = img.shape[0]
#     nrange = np.linspace(0, n - 1, n)

#     def vertical_cart_to_recto(a):
#         h = np.zeros([n, n], dtype=a.dtype)
#         for i in xrange(n):
#             indices = np.linspace(i, n - i - 1, n)
#             h[i] = interpolate.interp1d(nrange, a[i, :])(indices)
#         return h

#     h1 = vertical_cart_to_recto(img).T
#     h2 = vertical_cart_to_recto(img.T[::-1, :]).T
#     return np.r_[h1, h2]


# def recto_to_cart(img, method='linear'):
#     n = img.shape[1]
#     h = np.zeros([n, n], dtype=img.dtype)
#     for i in xrange(n / 2):
#         indices = np.linspace(i, n - i - 1, n)
#         cartrange = np.arange(i, n - i)
#         # top <-> left half-top
#         h[i, i:n - i] = interpolate.interp1d(indices, img[:n, i])(cartrange)
#         # right  <-> left half-bottom
#         h[i:n - i, n - i - 1] = interpolate.interp1d(
#             indices, img[n:2 * n, i])(cartrange)
#         #
#         h[n - i - 1, i:n - i] = interpolate.interp1d(
#             indices, img[:n, n - i - 1])(cartrange)[::-1]
#         h[i:n - i, i] = interpolate.interp1d(
#             indices, img[n:2 * n, n - i - 1])(cartrange)[::-1]
#     return h


# def caretesian_to_rectopolar_gridding_chunk(img, method='linear'):
#     cart = np.vstack(cartesian_coordinate_chunk(*img.shape))
#     fct = interpol_unstructured(cart, img.flatten(), method)
# #    return fct(polar_coordinate(*img.shape)).reshape(img.shape)
#     for chunk in rectopolar_coordinate_chunk(*img.shape):
#         yield fct(chunk)


# def rectopolar_to_cartesian_gridding_chunk(img, method='linear'):
#     new_shape = [(img.shape[0] + 1) / 2] * 2
# #    recto = polar_coordinate(*new_shape)
#     recto = np.vstack(rectopolar_coordinate_chunk(*new_shape))
# #    print "Recto to Cart: ", new_shape, recto.shape
#     fct = interpol_unstructured(recto, img.flatten(), method)
#     for chunk in cartesian_coordinate_chunk(*new_shape):
#         yield fct(chunk)


# def caretesian_to_rectopolar_gridding(img, method='linear'):
#     return np.vstack(caretesian_to_rectopolar_gridding_chunk(img, method))


# def rectopolar_to_cartesian_gridding(img, method='linear'):
#     return np.vstack(rectopolar_to_cartesian_gridding_chunk(img, method))


def stat(array):
    array = np.array(array)
    data = (array.shape, array.sum(), array.mean(),
            array.max(), array.min(), array.std(), np.percentile(array, 90))
    return "Shape:%s, Sum:%.3g, Mean:%.3g, Max:%.3g, Min:%.3g, Std:%.3g, P90:%.3g" % data


def robust_mean(array, k=1.5):
    if len(array) < 4:
        return array.mean()
    std = array.std()
    mean = array.mean()
    newarray = []
    for value in array:
        if value > mean - k * std and value < mean + k * std:
            newarray.append(value)
    return np.array(newarray).mean()


def partition(img, bx, by, ox=0, oy=0):
    # TODO: async version of that + seperate fct to calculate nblock
    blocks = []
    for x in range(0, img.shape[0] - ox, bx - ox):
        for y in range(0, img.shape[0] - oy, by - oy):
            blocks.append(img[x:x + bx, y:y + by])
    return blocks


def recompose(blocks, shape, bx, by, ox=0, oy=0):
    reconstructed = np.zeros(shape, dtype=blocks[0].dtype)
    lx = ly = 0
    for block in blocks:
        mask = np.ones_like(block) * 0.25
        if ly == 0:
            mask[:, :oy] += 0.25
        if lx == 0:
            mask[:ox, :] += 0.25
        if ly >= shape[0] - by:
            mask[:, oy:] += 0.25
        if lx >= shape[1] - bx:
            mask[ox:, :] += 0.25
        mask[mask == 0.75] = 1
        reconstructed[lx:lx + bx, ly:ly + by] += mask * block
        ly += by - oy
        if ly > shape[0] - by:
            ly = 0
            lx += bx - ox
    return reconstructed


def smooth(x, win_len, window_fct=np.hamming, boundary='zero', mode='valid'):
    if win_len <= 1:
        return x

    if x.ndim > 2 or x.ndim < 1:
        raise ValueError("Input signal should be of dimension 1 or 2")
    if np.array(x.shape).min() < win_len:
        raise ValueError("Input vector needs to be bigger than window size. %s vs %s" % (np.array(x.shape).min(), win_len))
    if window_fct is None:
        window_fct = lambda shape: np.ones(shape)
    if not hasattr(window_fct, '__call__'):
        raise ValueError("Input window fct should be a function")

    w = window_fct(get_next_odd(win_len))
    w = w / w.sum()

    return convolve(x, w, mode=mode, boundary=boundary)


def movingaverage(interval, window_size, boundary='zero', mode='valid'):
    window = np.ones(int(window_size))/float(window_size)
    return convolve(interval, window, mode=mode, boundary=boundary)


# def gaussian_window(n, alpha=2.5):
#     '''  '''
#     x = np.arange(-n / 2 + 1, n / 2 + 1)
#     k = gaussian_fct(0, 1, 0, n / (2 * alpha))(x)
#     return k / k.sum()


def gaussian_support(sigma=None, width=None, nsigma=4):
    if sigma is None:
        sigma = gaussian_fwhm_to_sigma(width)
    return int(np.ceil(2 * nsigma * sigma))


def gaussian_sigma_to_fwhm(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def gaussian_fwhm_to_sigma(fwhm):
    return 1 / (2. * np.sqrt(2 * np.log(2))) * fwhm


def get_pair(value, dtype=None):
    s = np.array(value, dtype=dtype)
    if s.ndim == 1:
        return s
    elif s.ndim == 0:
        return np.array([value, value], dtype=dtype)
    else:
        raise ValueError("Value should be of dimension 0 or 1 (is )" % s.ndim)


def is_callable(obj):
    return hasattr(obj, '__call__')


def is_str_number(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def is_number(x):
    return isinstance(x, (int, float, complex))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2floatlist(s):
    s = s.strip('([)]')
    return [float(k.strip()) for k in s.split(',')]


def str2jsonclass(v):
    if "py/type" not in v:
        return '{"py/type": "%s"}' % v
    return v


def str2jsonfunction(v):
    if "py/function" not in v:
        return '{"py/function": "%s"}' % v
    return v


def make_callable(obj):
    if not is_callable(obj):
        return lambda *args: obj
    return obj


def _get_next_oddeven(n, testeven):
    if isinstance(n, np.ndarray):
        if not n.dtype == int:
            n = n.astype(int)
        if (n % 2 == testeven).any():
            m = n.copy()
            m[(n % 2 == testeven)] += 1
            return m
        return n
    else:
        if not isinstance(n, int):
            n = int(n)
        if n % 2 == testeven:
            return n + 1
        return n


def get_next_even(n):
    return _get_next_oddeven(n, True)


def get_next_odd(n):
    return _get_next_oddeven(n, False)


def gaussian_moments(data):
    '''Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments '''
    height = data.max()
    center = np.zeros(data.ndim)
    width = np.zeros(data.ndim)

    total = data.sum()
    indices = np.indices(data.shape)

    for i in range(data.ndim):
        center[i] = (indices[i] * data).sum() / total

    for i in range(data.ndim):
        index = [slice(np.around(k), np.around(k) + 1) for k in center]
        index[i] = slice(None)
        line = np.ravel(data[index])
        try:
            width[i] = np.sqrt(
                abs((np.arange(line.size) - center[i - 1]) ** 2 * line).sum() / line.sum())
        except:
            pass

    return height, center, width


def gaussian_fct(a, height, center, sigma, angle=0):
    '''Returns a gaussian function with the given parameters, angle in radian'''

    def gaussian(indices):
        if is_number(center):
            u = (center - indices) ** 2 / (2. * sigma ** 2)
        elif len(center) == 1:
            u = (center[0] - indices) ** 2 / (2. * sigma[0] ** 2)
        elif len(center) == 2:
            x, y = indices
            center_x, center_y = center
            sigmax, sigmay = sigma
            xp = (x - center_x) * np.cos(angle) - (y - center_y) * np.sin(angle)
            yp = (x - center_x) * np.sin(angle) + (y - center_y) * np.cos(angle)
            u = xp ** 2 / (2. * sigmax ** 2) + yp ** 2 / (2. * sigmay ** 2)
        return a + height * np.exp(-u)

    return gaussian


def lorentzian_1d(a, height, center, gamma):
    return lambda x: a + height * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def fitgaussian(data, params, base_null=True):
    ''' params: (a, height, center, sigma)
        center and sigma can be array or number

        Return fitted params, cov'''
    if params is None:
        return None

    center_is_array = not is_number(params[2])

    def flat_params(params):
        if not center_is_array:
            return params
        flat_params = []
        for param in list(params):
            if is_number(param):
                flat_params.append(param)
            else:
                flat_params.extend(param)
        return flat_params

    def unflat_params(params):
        if not center_is_array:
            return params
        if base_null:
            params[0] = 0
        print(params)
        return [params[0], params[1], params[2:2 + data.ndim], params[2 + data.ndim: 2 + 2 * data.ndim]]

    def fct(p):
        return np.ravel(gaussian_fct(*unflat_params(p))(np.indices(data.shape)) - data)

    res, cov = optimize.leastsq(fct, flat_params(params))

    return unflat_params(res), cov



# def bounded_leastq(func, params, LB=None, UB=None, boundweight=10, **kwargs):

#     def bound_constraints(p):
#         return boundweight * [(p[i] < LB[i]) + (p[i] > UB[i]) for i in range(len(p))]

#     def errorfunction(p):
#         err = func(*p)
#         err = np.hstack([err, bound_constraints(p)])
#         return err

#     return optimize.leastsq(errorfunction, params**kwargs)


# def modelfit(data):

#     def errorfunction(*p):
#         indices = np.indices(np.array(data.shape, dtype=np.int))
#         p = [0, p[2], p[0], p[1], p[3], p[3], 0]

#         err = np.ravel(data - gaussian_fct(*p)(indices))

#         return err

#     c0x, c0y = coord_max(data)
#     params = [c0x, c0y, data[c0x, c0y], 1]
#     LB = [0, 0, 0, 0]
#     UB = [data.shape[0], data.shape[1], 100, max(data.shape)]

#     p = bounded_leastq(errorfunction, params, LB=LB, UB=UB)

#     return p


def pdist(X, metric_fct):
    n = len(X)

    dm = np.zeros((n * (n - 1) / 2,), dtype=np.double)
    k = 0
    for i in xrange(0, n - 1):
        for j in xrange(i + 1, n):
            dm[k] = metric_fct(X[i], X[j])
            k = k + 1
    return dm


def cluster(x, z, t, **kargs):
    from scipy.cluster.hierarchy import fcluster

    c = fcluster(z, t, **kargs)
    res = [[] for k in range(max(c))]
    for i, j in enumerate(c):
        res[j - 1].append(x[i])
    return res


def assert_close(a, b, digit=7):
    assert round(a - b, digit) == 0


def assert_equal(first, second, rtol=1e-5, atol=1e-8):
    first = np.array(first)
    second = np.array(second)
    if first.shape != second.shape:
        raise Exception("Shape differ")
    try:
        np.allclose(first, second, rtol=rtol, atol=atol)
        return True
    except:
        pass
    raise Exception("Shape differ")


def assert_raise(exception, fct, *args):
    try:
        fct(*args)
        assert False
    except exception:
        assert True


class AbstractParameter(object):

    def __call__(self):
        return self.get()

    def get(self, shape=None):
        pass


class Constant(AbstractParameter):

    def __init__(self, value):
        self.value = value        

    def get(self, shape=None):
        if shape is None:
            return self.value
        a = np.empty(shape)
        a.fill(self.value)
        return a


class AbstractRandomUniform(AbstractParameter):

    def __init__(self, min, max, seed=None):
        self.max = max
        self.min = min
        self.seed = seed


class RandomUniformInt(AbstractRandomUniform):

    def get(self, shape=None):
        if self.min == self.max:
            return Constant(self.min).get(shape)
        return get_random(seed=self.seed).randint(self.min, self.max, size=shape)


class RandomUniformFloat(AbstractRandomUniform):

    def get(self, shape=None):
        if self.min == self.max:
            return Constant(self.min).get(shape)
        return get_random(seed=self.seed).uniform(self.min, self.max, size=shape)


class RandomNormal(AbstractParameter):

    def __init__(self, mean, sigma, seed=None):
        self.mean = mean
        self.sigma = sigma
        self.seed = seed

    def get(self, shape=None):
        if self.sigma == 0:
            return Constant(self.mean).get(shape)
        return get_random(seed=self.seed).normal(self.mean, self.sigma, size=shape)


class UniformFloat(AbstractParameter):

    def __init__(self, min, max, seed=None):
        self.max = max
        self.min = min
        self.i = 1
        self.nstep = 0.

    def get(self):
        if self.nstep == 0:
            self.nstep = 1
            return self.min
        value = self.min + self.i * (self.max - self.min) / float(self.nstep)
        self.i += 2
        if self.i >= self.nstep:
            self.nstep *= 2
            self.i = 1
        return value


class ConfigurationsContainer(object):

    def __init__(self, configs):
        self._configs = configs

    def __str__(self):
        return self.values()

    def add_config(self, config):
        self._configs.append(config)

    def to_file(self, filename):
        parser = configparser.RawConfigParser()
        for config in self._configs:
            section =  config.get_title()
            parser.add_section(section)
            for option, value in config.items(max_level=1, encode=True):
                parser.set(section, option, value)

        with open(filename, 'wb') as fh:
            parser.write(fh)

    def from_file(self, filename):
        parser = configparser.RawConfigParser()
        parser.read(filename)
        configs = dict([(config.get_title(), config) for config in self._configs])
        for section in parser.sections():
            config = configs[section]
            for option, value in parser.items(section):
                config.set(option, value, decode=True)

    def doc(self, max_level=0):
        return "\n".join([k.doc(max_level=max_level) for k in self._configs])

    def values(self, max_level=0):
        return "\n".join([k.values(max_level=max_level) for k in self._configs])



class BaseConfiguration(ConfigurationsContainer):

    def __init__(self, settings, title):
        # settings order: key, default, doc, validator, decoder, encoder, level
        # class attributes need to start with "_" so that we can set 
        # option using simple attribute assignment (ex: config.option = value)
        # levels: 0: show in doc(), 1: save/load to/from file, 2: not saved
        self._title = title
        self._keys, defaults, docs, validators, decoders, encoders, level = zip(*settings)
        self._defaults = collections.OrderedDict(zip(self._keys, defaults))
        self._values = collections.OrderedDict(zip(self._keys, defaults))
        self._docs = collections.OrderedDict(zip(self._keys, docs))
        self._validators = collections.OrderedDict(zip(self._keys, validators))
        self._decoders = collections.OrderedDict(zip(self._keys, decoders))
        self._encoders = collections.OrderedDict(zip(self._keys, encoders))
        self._level = collections.OrderedDict(zip(self._keys, level))
        ConfigurationsContainer.__init__(self, [self])

    def __getattr__(self, name):
        if name in self._keys:
            return self.get(name)
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name[0] != "_":
            if name in self._keys:
                self.set(name, value)
            else:
                print("Warning: No option with name", name)
        else:
            object.__setattr__(self, name, value)

    def __str__(self):
        return self.values()

    def get(self, option, encode=False):
        assert option in self._keys
        value = self._values.get(option)
        if encode and value is not None:
            return self._encoders.get(option, str)(value)
        return value

    def has(self, option):
        return option in self._keys

    def iter_options(self, max_level=2):
        for key in self._keys:
            if self._level.get(key) <= max_level:
                yield key

    def set(self, option, value, validate=True, decode=False):
        assert option in self._keys
        decoders = self._decoders.get(option)
        validator = self._validators.get(option)
        if decode:
            if value == 'None':
                value = None
            elif decoders is not None:
                value = decoders(value)
        if value is not None and validate and validator is not None:
            if not validator(value):
                print("Warning: validation failed while setting", (option, value))
        self._values[option] = value

    def items(self, max_level=2, encode=False):
        for key in self.iter_options(max_level=max_level):
            yield (key, self.get(key, encode=encode))

    def get_default(self, option):
        assert option in self._keys
        return self._defaults.get(option)

    def reset(self, option):
        assert option in self._keys
        self.set(option, self.get_default(option))

    def reset_all(self):
        for key in self._keys:
            self.reset(key)

    def get_doc(self, option):
        assert option in self._keys
        return self._docs.get(option)

    def doc(self, max_level=0):
        array = []
        for key in self.iter_options(max_level=max_level):
            doc = self.get_doc(key)
            default = self.get_default(key)
            array.append([key, doc, default])

        table = format_table(array, ["Option", "Documentation", "Default"], max_col_size=40)

        return "Documentation for:%s\n%s" % (self._title, table)

    def values(self, max_level=0):
        array = []
        for key in self.iter_options(max_level=max_level):
            value = self.get(key)
            array.append([key, value])

        table = format_table(array, ["Option", "Value"], max_col_size=40)

        return "%s:\n%s" % (self._title, table)

    def get_title(self):
        return self._title


def validator_in_range(vmin, vmax, instance=(float, int)):

    def validator(value):
        if isinstance(value, instance) and value >= vmin and value <= vmax:
            return True
        return False

    return validator


def validator_in(list_allowed):

    def validator(value):
        if value in list_allowed:
            return True
        return False

    return validator


def validator_is(instance):

    def validator(value):
        return isinstance(value, instance)

    return validator


def validator_is_class(klass):

    def validator(value):
        return issubclass(value, klass)

    return validator


def validator_list(length, instance=None):

    def validator(l):
        if not isinstance(l, (list, np.ndarray)):
            return False
        if len(l) != length:
            return False
        for value in l:
            if not isinstance(value, instance):
                return False
        return True

    return validator


def format_table(data, header=None, min_col_size=10, max_col_size=None):
    ''' Return a table formatted version of data'''

    dim = len(data[0])
    if header :
        assert len(header) == dim 
        data = [header] + [None] + data
    col_size = [min_col_size] * dim
    for i, line in enumerate(data) :
        if line is None:
            # header delimitation
            continue
        assert len(line) == dim, "Dimension incorrect (%s != %s)" % (len(line), dim)
        new_line = [""] * dim
        for j in range(dim) :
            s = str(line[j]).strip()
            if max_col_size is not None and len(s) > max_col_size:
                iws = s.rfind(" ", 0, max_col_size)
                if iws != -1 and iws > 0:
                    s, new_line[j] = s[:iws].strip(), s[iws:].strip()
            n = len(s) + 1
            data[i][j] = s
            if n > col_size[j] :
                col_size[j] = n
        if sum([len(k) for k in new_line]) > 0:
            data.insert(i + 1, new_line)
    res = ""
    for line in data:
        if line is None:
            res += "-" * (sum(col_size) + dim) + "\n"
            continue
        for i in range(dim) :
            res += "%-*s" % (col_size[i], str(line[i])[:col_size[i]])
            res += '|'
        res += "\n" 
    return res

class AbstractFct(object):

    def __init__(self, p0):
        self.p0 = p0

    @staticmethod
    def fct(x, p):
        pass

    def __call__(self, x):
        return self.fct(x, *self.p0)

    def get_text_equ(self, label=''):
        return "None"

    def error(self, x, y):
        return (self(x) - y).std(ddof=1) / np.sqrt(len(x))

    @staticmethod
    def fit(x, y):
        pass


class LinearFct(AbstractFct):

    def __init__(self, a, b, ea=None, eb=None):
        self.a = a
        self.b = b
        self.ea = ea
        self.eb = eb
        AbstractFct.__init__(self, [a, b])

    @staticmethod
    def fct(x, a, b):
        return a * np.asarray(x) + b

    @staticmethod
    def from_angle_point(angle, point):
        return LinearFct(*affine_fct_from_angle_point(angle, point))

    def inverse_fct(self, y, a, b):
        if a != 0:
            return (y - b) / a
        return 0

    def get_text_equ(self, label='y'):
        return "$%s = %.5f x + %.2f$" % (label, self.a, self.b)

    @staticmethod
    def fit(x, y, sigma=None):
        x = np.asarray(x)
        y = np.asarray(y)

        if np.__version__ < 1.5 and sigma is not None:
            errfunc = lambda p, l, u: (u - LinearFct.fct(l, *p)) / err
            pinit = [1.0, 1.0]
            (a, b), success = leastsq(errfunc, pinit, args=(x, y))
        else:
            w = None
            if sigma is not None:
                w = 1 / np.array(sigma)
            b, a = np.polynomial.polynomial.polyfit(x, y, 1, w=w)

        fct = LinearFct(a, b)
        RMSE = (fct(x) - y).std(ddof=2)
        Sxx = (x ** 2).sum() - len(x) * x.mean() ** 2
        ea = RMSE / np.sqrt(Sxx)
        eb = RMSE * np.sqrt(1 / len(x) + x.mean() ** 2 / Sxx)

        return LinearFct(a, b, ea=ea, eb=eb)


class PowerFct1(AbstractFct):

    def __init__(self, a, b, ea=None, eb=None):
        self.a = a
        self.b = b
        self.ea = ea
        self.eb = eb
        AbstractFct.__init__(self, [a, b])

    @staticmethod
    def fct(x, a, b):
        return b * np.asarray(x) ** a

    @staticmethod
    def fit(x, y, sigma=None):
        logx = np.log(x)
        logy = np.log(y)
        linfct = LinearFct.fit(logx, logy)
        return PowerFct1(linfct.a, np.exp(linfct.b), ea=linfct.ea, eb=np.abs(1 / linfct.b) * linfct.eb)


class AbsLinearFct(LinearFct):

    @staticmethod
    def fct(x, a, b):
        return np.abs(a * np.asarray(x) + b)


class CosinusFct(AbstractFct):

    def __init__(self, a, w):
        self.a = a
        self.w = w
        AbstractFct.__init__(self, [a, w])

    @staticmethod
    def fct(x, a, w):
        return a * np.cos(x * w)

    def get_text_equ(self, label='y'):
        return "$%s = %.2f \cos(%.4f x)$" % (label, self.a, self.w)

    @classmethod
    def fit(klass, x, y):

        def errfunc(p, x, y):
            err = klass.fct(x, *p) - y
            return err

        guess_a = np.mean(y)
        guess_b = 3 * np.std(y) / (2 ** 0.5)
        p0 = (guess_a, guess_b)
        p0 = (self.a, self.w)
        try:
            p1, success = leastsq(errfunc, p0, args=(x, y))
            return klass(p1[0], p1[1])
        except:
            return klass(p0[0], p0[1])


class SinusFct(AbstractFct):

    def __init__(self, a, r, w, phi):
        self.a = a
        self.r = r
        self.phi = phi
        self.w = w
        AbstractFct.__init__(self, [a, r, w, phi])

    @staticmethod
    def fct(x, a, r, w, phi):
        return a + r * np.sin(x * w + phi)

    def fit(self, x, y):

        def errfunc(p, x, y):
            err = SinusFct.fct(x, *p) - y
            return err

        try:
            p1, success = leastsq(errfunc, self.p0, args=(np.array(x), np.array(y)))
            return SinusFct(*p1)
        except Exception:
            return SinusFct(*self.p0)


class AbsCosinusFct(CosinusFct):

    def fct(self, x, a, w):
        return a * np.abs(np.cos(x * w))

    def get_text_equ(self, label='y'):
        return "$%s = %.2f abs(\cos(%.4f x * 2 \pi))$" % (label, self.a, self.w)


class GeneralisedLogisticFct(AbstractFct):
    ''' http://en.wikipedia.org/wiki/Generalized_logistic_curve 

        A: the lower asymptote;
        K: the upper asymptote. If A=0 then K is called the carrying capacity;
        B: the growth rate;
        nu>0 : affects near which asymptote maximum growth occurs.
        Q: depends on the value Y(0)
        M: the time of maximum growth if Q=nu'''

    def __init__(self, a, k, b, nu, q, m):
        self.a = a
        self.k = k
        self.b = b
        self.nu = nu
        self.q = q
        self.m = m
        AbstractFct.__init__(self, [a, k, b, nu, q, m])

    @staticmethod
    def fct(x, a, k, b, nu, q, m):
        return a + float(k - a) / (1. + float(q) * np.exp(-b * (x - m))) ** (1 / float(nu))

    def inverse_fct(self, y, a, k, b, nu, q, m):
        return - np.log((((k - a) / float(y - a)) ** nu - 1) / float(q)) / float(b) + m

    def fit(self, x, y):
        popt, pcov = curve_fit(self.fct, x, y, p0=self.p0)

        return self.__class__(*popt)


class InverseFct(AbstractFct):

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        AbstractFct.__init__(self, [a, b, c, d])

    def fct(self, x, a, b, c, d):
        return float(a) + float(b) / (float(c) + x) ** float(d)

    def fit(self, x, y):
        popt, pcov = curve_fit(self.fct, x, y, p0=self.p0)

        return self.__class__(*popt)


class PolynomialFct(AbstractFct):

    def __init__(self, *args):
        " in increasing power "
        self.p0 = args
        AbstractFct.__init__(self, args)

    def fct(self, x, *args):
        return np.poly1d(args[::-1])(x)

    def get_text_equ(self, label='y'):
        s = '$%s = ' % label
        for i, a in enumerate(self.p0):
            s += "+%.4f x^%i" % (a, i)
        s += "$"
        return s

    def fit(self, x, y):
        p = np.polyfit(x, y, len(self.p0))
        return PolynomialFct(*p[::-1])


class RationalFct(AbstractFct):

    def __init__(self, numerateur, denominateur):
        self.numerateur = numerateur
        self.denominateur = denominateur
        AbstractFct.__init__(self, numerateur.p0 + denominateur.p0)

    def fct(self, x, *args):
        num_args = args[:len(self.numerateur.p0)]
        den_args = args[len(self.numerateur.p0):]
        return self.numerateur.fct(x, *num_args) / self.denominateur.fct(x, *den_args)

    def fit(self, x, y):
        popt, pcov = curve_fit(self.fct, x, y, p0=self.p0)

        num_args = popt[:len(self.numerateur.p0)]
        num_poly = PolynomialFct(*num_args)
        den_args = popt[len(self.numerateur.p0):]
        den_poly = PolynomialFct(*den_args)

        return self.__class__(num_poly, den_poly)


class ExponentielFct(AbstractFct):

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        AbstractFct.__init__(self, [a, b, c])

    def fct(self, x, a, b, c):
        return a * np.exp(b * x) + c

    def fit(self, x, y):
        popt, pcov = curve_fit(self.fct, x, y, p0=self.p0)

        return self.__class__(*popt)


class PowerFct(AbstractFct):

    def __init__(self, a, b, c, r):
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        AbstractFct.__init__(self, [a, b, c, r])

    def fct(self, x, a, b, c, r):
        return a + c * (x + b) ** r

    def fit(self, x, y):
        popt, pcov = curve_fit(self.fct, x, y, p0=self.p0)

        return self.__class__(*popt)


class AbstractFilter:

    def __or__(self, other):
        return FilterOr(self, other)

    def __and__(self, other):
        return FilterAnd(self, other)

    def filter(self, *args):
        raise NotImplementedError()


class FilterAnd(AbstractFilter):

    def __init__(self, a, b):
        self.__a = a
        self.__b = b

    def __str__(self):
        return "(%s and %s)" % (str(self.__a), str(self.__b))

    def filter(self, *args):
        return self.__a.filter(*args) and self.__b.filter(*args)


class FilterOr(AbstractFilter):

    def __init__(self, a, b):
        self.__a = a
        self.__b = b

    def __str__(self):
        return "(%s or %s)" % (str(self.__a), str(self.__b))

    def filter(self, *args):
        return self.__a.filter(*args) or self.__b.filter(*args)


class DummyFilter(AbstractFilter):

    def filter(self, filter):
        return True


def test_upsample():
    import time
    import plotutils, imgutils
    from scipy.misc import lena
    from matplotlib.mlab import csd, detrend_mean
    from scipy.ndimage.interpolation import rotate
    from scipy.spatial.distance import cdist

    i1 = imgutils.gaussian(50, width=10, center=[25.2, 25])
    i2 = imgutils.gaussian(50, width=10, center=[25, 25])

    def upsamplefft(X, n):
        shape = np.array(X.shape)
        print(n, shape, np.fft.ifftshift(resize(np.fft.fftshift(X), n * shape, padding_mode='center')).shape)
        return np.fft.ifftshift(resize(np.fft.fftshift(X), n * shape, padding_mode='center'))

    def fourier_interp1d(data, out_x, data_x=None, nthreads=1, use_numpy_fft=False,
            return_real=True):
        # specify fourier frequencies
        freq = np.fft.fftfreq(data.size)[:,np.newaxis]

        # reshape outinds
        if out_x.ndim != 1:
            raise ValueError("Must specify a 1-d array of output indices")

        if data_x is not None:
            if data_x.shape != data.shape:
                raise ValueError("Incorrect shape for data_x")
            # interpolate output indices onto linear grid
            outinds = np.interp(out_x, data_x, np.arange(data.size))[np.newaxis,:]
        else:
            outinds = out_x[np.newaxis,:]

        # create the fourier kernel
        kern=np.exp((1j*2*np.pi)*freq*outinds)

        # the result is the dot product (sum along one axis) of the inverse fft of
        # the function and the kernel
        result = np.dot(np.fft.fftn(data),kern)

        if return_real:
            return result.real
        else:
            return result

    def dftups(inp,nor=None,noc=None,usfac=1,roff=0,coff=0):
        '''
        *translated from matlab*
        http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

        Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
        a small region.
        usfac Upsampling factor (default usfac = 1)
        [nor,noc] Number of pixels in the output upsampled DFT, in
        units of upsampled pixels (default = size(in))
        roff, coff Row and column offsets, allow to shift the output array to
        a region of interest on the DFT (default = 0)
        Recieves DC in upper left corner, image center must be in (1,1)
        Manuel Guizar - Dec 13, 2007
        Modified from dftus, by J.R. Fienup 7/31/06

        This code is intended to provide the same result as if the following
        operations were performed
        - Embed the array "in" in an array that is usfac times larger in each
        dimension. ifftshift to bring the center of the image to (1,1).
        - Take the FFT of the larger array
        - Extract an [nor, noc] region of the result. Starting with the
        [roff+1 coff+1] element.

        It achieves this result by computing the DFT in the output array without
        the need to zeropad. Much faster and memory efficient than the
        zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
        '''
        # this function is translated from matlab, so I'm just going to pretend
        # it is matlab/pylab
        from numpy.fft import ifftshift,fftfreq
        from numpy import pi,newaxis,floor

        nr,nc=np.shape(inp);
        # Set defaults
        if noc is None: noc=nc;
        if nor is None: nor=nr;
        # Compute kernels and obtain DFT by matrix products
        term1c = ( ifftshift(np.arange(nc,dtype='float') - floor(nc/2)).T[:,newaxis] )/nc # fftfreq
        term2c = (( np.arange(noc,dtype='float') - coff )/usfac)[newaxis,:] # output points
        kernc=np.exp((-1j*2*pi)*term1c*term2c);

        term1r = ( np.arange(nor,dtype='float').T - roff )[:,newaxis] # output points
        term2r = ( ifftshift(np.arange(nr,dtype='float')) - floor(nr/2) )[newaxis,:] # fftfreq
        kernr=np.exp((-1j*2*pi/(nr*usfac))*term1r*term2r);
        #kernc=exp((-i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
        #kernr=exp((-i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2) ));
        out=np.dot(np.dot(kernr,inp),kernc);
        #return np.roll(np.roll(out,-1,axis=0),-1,axis=1)
        return out 


    def fourier_interp2d(data, outinds, nthreads=1, use_numpy_fft=False,
            return_real=True):

        # load fft
        fftn,ifftn = np.fft.fftn, np.fft.ifftn

        if hasattr(outinds,'ndim') and outinds.ndim not in (data.ndim+1,data.ndim):
            raise ValueError("Must specify an array of output indices with # of dimensions = input # of dims + 1")
        elif len(outinds) != data.ndim:
            raise ValueError("outind array must have an axis for each dimension")

        imfft = ifftn(data)

        freqY = np.fft.fftfreq(data.shape[0])
        if hasattr(outinds,'ndim') and outinds.ndim == 3:
            # if outinds = np.indices(shape), we extract just lines along each index
            indsY = freqY[np.newaxis,:]*outinds[0,:,0][:,np.newaxis]
        else:
            indsY = freqY[np.newaxis,:]*np.array(outinds[0])[:,np.newaxis]
        kerny=np.exp((-1j*2*np.pi)*indsY)

        freqX = np.fft.fftfreq(data.shape[1])
        if hasattr(outinds,'ndim') and outinds.ndim == 3:
            # if outinds = np.indices(shape), we extract just lines along each index
            indsX = freqX[:,np.newaxis]*outinds[1,0,:][np.newaxis,:]
        else:
            indsX = freqX[:,np.newaxis]*np.array(outinds[1])[np.newaxis,:]
        kernx=np.exp((-1j*2*np.pi)*indsX)

        result = np.dot(np.dot(kerny, imfft), kernx)

        if return_real:
            return result.real
        else:
            return result

    x = np.arange(100)
    y = 2 + 0.1 * x + 0.002 * x ** 2

    x2 = np.linspace(0, 100, 200)
    y2 = fourier_interp1d(y, x2) / 100.

    M = np.array(i1.shape) + np.array(i2.shape) - 1
    n = 10
    X = np.fft.rfftn(i1, M)
    Y = np.fft.rfftn(flip(i2), M)
    X = upsamplefft(X, n)
    Y = upsamplefft(Y, n)
    res = np.fft.irfftn(X * Y)
    res = resize(res, n * M, 'left')
    res = res[:-n, :-n]

    xcorr = xcorr_fast(i1, i2, mode='all')
    xcorr = norm_xcorr2(i1, i2)
    # xcorr = xcorr[:-1, :-1]

    # print np.array(xcorr.shape) / 2. - coord_max(xcorr)
    # print np.array(res.shape), coord_max(res), np.array(res.shape) / 2. - coord_max(res), (np.array(res.shape) / 2. - coord_max(res)) / float(n)

    stack = plotutils.FigureStack()
    fig, (ax1, ax2, ax3, ax4) = stack.add_subplots("Test", n=4, reshape=False)
    ax1.plot(x, y)
    ax1.plot(x2, y2)
    ax1.plot(x2, 2 * np.fft.ifftn(upsamplefft(np.fft.fftn(y), 2)))
    ax2.imshow(i1)
    ax3.imshow(np.fft.irfftn(np.fft.rfftn(i1), [100, 100]))
    # ax4.imshow(fourier_interp2d(i1, (np.arange(100), np.arange(100))))
    # ax4.imshow(4 * np.fft.irfftn(upsamplefft(np.fft.rfftn(i1), 2)))
    ax4.imshow(dftups(i1, 100, 100, 1).real)

    fig, (ax1, ax2, ax3, ax4) = stack.add_subplots("Test", n=4, reshape=False)
    ax1.imshow(i1)
    ax2.imshow(i2)
    ax3.imshow(xcorr)
    ax4.imshow(cdist(i2, i1, 'correlation'))

    fig, (ax1, ax2, ax3, ax4) = stack.add_subplots("Test", n=4, reshape=False)
    ax1.imshow(X.real)
    ax2.imshow(Y.real)
    ax3.imshow((X * Y).real)
    ax4.imshow(res.real)
    stack.show()


def test_zero_ssd():
    from libwise import imgutils, plotutils
    x = imgutils.gaussian(50, width=5, center=[20, 20]) * 1.5
    y = imgutils.gaussian(50, width=5, center=[25, 25])

    mode = 'same'
    method = 'auto'

    ny = local_sum(np.ones_like(x), y.shape, mode=mode)
    x_mean = local_sum(x, y.shape, mode=mode) / ny
    y_mean = xcorr_fast(np.ones_like(x), y, mode=mode, method=method) / ny

    x = x - x_mean
    y = y - y_mean

    xcorr = xcorr_fast(x, y, mode=mode, method=method)

    local_sum_x2 = local_sum(x ** 2, y.shape, mode=mode)

    ysum2 = xcorr_fast(np.ones_like(x), y ** 2, mode=mode, method=method)

    ssd = local_sum_x2 + ysum2 - 2. * xcorr

    stack = plotutils.FigureStack()
    fig, axs = stack.add_subplots("test", n=6, reshape=False)

    axs[0].imshow(x_mean)
    axs[1].imshow(y_mean)
    axs[2].imshow(local_sum_x2)
    axs[3].imshow(ysum2)
    axs[4].imshow(xcorr)
    axs[5].imshow(ssd)

    stack.show()


def test_find_peak():
    from libwise import imgutils, plotutils

    stack = plotutils.FigureStack()
    fig, axs = stack.add_subplots("test", n=2, reshape=False)

    noise = gaussian_noise([50, 50], 0, 0.01)
    img = imgutils.gaussian(50, width=10, center=[45.4, 25.2]) + noise
    img += imgutils.gaussian(50, width=10, center=[35.4, 25.2])

    img_peak, peaks = find_peaks(img, 4, 0.4, fit_gaussian=True, fit_gaussian_n=2)

    axs[0].imshow(img)
    axs[1].imshow(img_peak)
    stack.show()


def test_permutations():
    a = [1, 2, 3, 4, 5]
    a = ['a', 'b', 'c', 'd', 'e']
    print(permutation_no_succesive(a))


def test_local_max():
    from libwise import imgutils, plotutils

    heights = np.random.uniform(5, 10, (3))
    widths = np.random.uniform(5, 10, (3, 2))
    center = np.random.uniform(10, 90, (3, 2))

    img = imgutils.multiple_gaussian([100, 100], heights, widths, center)
    img += np.random.normal(0, 1, img.shape)

    for center, height in zip(center, heights):
        search_center = np.ceil(np.random.normal(np.round(center), [2, 2]))
        print(center, height, search_center)
        print(local_max(img, search_center, 5, fit_gaussian=True))

    stack = plotutils.FigureStack()
    fig, ax = stack.add_subplots("test")

    ax.imshow(img)

    stack.show()


if __name__ == '__main__':
    test_local_max()
