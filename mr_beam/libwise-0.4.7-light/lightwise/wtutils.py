'''
Created on Feb 13, 2012

@author: fmertens

This is a python3 ported version of Florent Mertens WISE code: https://github.com/flomertens/wise
'''

import numpy as np

import lightwise.nputils as nputils
import lightwise.imgutils as imgutils
import lightwise.wavelets as wavelets


def get_wavelet_obj(w):
    if isinstance(w, str):
        return wavelets.get_wavelet(w)
    if isinstance(w, wavelets.WaveletBase):
        return w
    raise ValueError("w is not a correct wavelet")


def dwt(signal, wavelet, boundary, level=None, initial_signal=None, axis=None):
    '''
    Perform a one level discrete wavelet transform.

    Result is len k + l + 1 if len(s) = 2k  and len(hkd) = 2l
        it is len k + l if len(s) = 2k + 1

    '''
    hkd = get_wavelet_obj(wavelet).get_dec_hk()
    gkd = get_wavelet_obj(wavelet).get_dec_gk()

    a_conv = nputils.convolve(signal, hkd, boundary, axis=axis)
    d_conv = nputils.convolve(signal, gkd, boundary, axis=axis)

    a = nputils.downsample(a_conv, 2, oddeven=1, axis=axis)
    d = nputils.downsample(d_conv, 2, oddeven=1, axis=axis)

    return (a, d)


def dwt_inv(a, d, wavelet, boundary, level=None, axis=None):
    '''
    Perform a one level inverse discrete wavelet transform.

    Result len is always 2 * len(a) - len(hkr) + 1

    Warning: if len(s) = 2k + 1, then idwt(dwt(s)) will give one element
             too much which will be zero. There is no way to know the
             parity of the original signal. It can be safely removed.
             For this reason, if len(a) is bigger than len(d) to 1, we strip
             this last element
    '''
    if len(a) == len(d) + 1:
        a = a[:-1]

    hkr = get_wavelet_obj(wavelet).get_rec_hk()
    gkr = get_wavelet_obj(wavelet).get_rec_gk()

    a_upsample = nputils.upsample(a, 2, oddeven=1, lastzero=True, axis=axis)
    d_upsample = nputils.upsample(d, 2, oddeven=1, lastzero=True, axis=axis)

    c1 = nputils.convolve(a_upsample, hkr, boundary, axis=axis, mode='valid')
    c2 = nputils.convolve(d_upsample, gkr, boundary, axis=axis, mode='valid')

    return c1 + c2


def uwt(signal, wavelet, boundary, level, initial_signal=None, axis=None):
    hkd = nputils.atrou(get_wavelet_obj(wavelet).get_dec_hk(), pow(2, level))
    gkd = nputils.atrou(get_wavelet_obj(wavelet).get_dec_gk(), pow(2, level))

    a = nputils.convolve(signal, hkd, boundary, axis=axis)
    d = nputils.convolve(signal, gkd, boundary, axis=axis)

    return (a, d)


def uwt_inv(a, d, wavelet, boundary, level, initial_signal=None, axis=None):
    hkr = nputils.atrou(get_wavelet_obj(wavelet).get_rec_hk(), pow(2, level))
    gkr = nputils.atrou(get_wavelet_obj(wavelet).get_rec_gk(), pow(2, level))

    c1 = nputils.convolve(a, hkr, boundary, axis=axis, mode="valid")
    c2 = nputils.convolve(d, gkr, boundary, axis=axis, mode="valid")

    return 1 / 2. * (c1 + c2)


def uiwt(signal, wavelet, boundary, level, initial_signal=None, axis=None):
    hkd = nputils.atrou(get_wavelet_obj(wavelet).get_dec_hk(), pow(2, level))

    a = nputils.convolve(signal, hkd, boundary, axis=axis, mode='same')

    d = signal - a

    return (a, d)


def uimwt(signal, wavelet, boundary, level, initial_signal=None, axis=None):
    hkd = nputils.atrou(get_wavelet_obj(wavelet).get_dec_hk(), pow(2, level))

    a = nputils.convolve(signal, hkd, boundary, axis=axis, mode='same')
    a2 = nputils.convolve(a, hkd, boundary, axis=axis, mode='same')

    d = signal - a2

    return (a, d)


def uiwt_inv(a, d, wavelet, boundary, level, axis=None):
    return a + d


def wavedec(signal, wavelet, level, boundary="symm",
            dec=dwt, axis=None, thread=None):
    # max_level = get_wavelet_obj(wavelet).get_max_level(signal)
    # if level > max_level:
        # raise ValueError("Level should be < %s" % max_level)
    res = []
    a = signal
    for j in range(int(level)):
        if thread and not thread.is_alive():
            return None
        a, d = dec(a, wavelet, boundary, j, initial_signal=signal, axis=axis)
        res.append(d)
    res.append(a)
    return res


def dogdec(signal, widths=None, angle=0, ellipticity=1, boundary="symm"):
    if widths is None:
        widths = np.arange(1, min(signal.shape) / 4)
    beams =  [imgutils.GaussianBeam(ellipticity * w, w, bpa=angle) for w in widths]
    filtered = [b.convolve(signal, boundary=boundary) for b in beams]
    res = [(el[0] - el[-1]) for el in nputils.nwise(filtered, 2)]
    for s in res:
        s[s <= 0] = 0
    res = [s - b2.convolve(s, boundary=boundary) for (s, (b1, b2)) in zip(res, nputils.nwise(beams, 2))]
    # res = [b1.convolve(s, boundary=boundary) - b2.convolve(s, boundary=boundary) for (s, (b1, b2)) in zip(res, nputils.nwise(beams, 2))]
    return res


def pyramiddec(signal, widths=None, angle=0, ellipticity=1, boundary="symm"):
    if widths is None:
        widths = np.arange(1, min(signal.shape) / 4)
    beams =  [imgutils.GaussianBeam(ellipticity * w, w, angle=angle) for w in widths]
    min_scale = beams[0].convolve(signal, boundary=boundary) - beams[1].convolve(signal, boundary=boundary)
    filtered_min = [b.convolve(min_scale, boundary=boundary) for b in beams]
    filtered_all = [b.convolve(signal, boundary=boundary) for b in beams]
    dog = [(el[0] - el[-1]) for el in nputils.nwise(filtered_all, 2)]
    return [v - k for k, v in  zip(filtered_min, dog)]


def waverec(coefs, wavelet, boundary="symm", rec=dwt_inv,
            axis=None, shape=None, thread=None):
    a = coefs[-1]
    for j in range(len(coefs) - 2, -1, -1):
        if thread and not thread.is_alive():
            return None
        a = rec(a, coefs[j], wavelet, boundary, j, axis=axis)
    if shape and shape != a.shape:
        # See idwt() for an explaination
        a = nputils.index(a, np.s_[:-1], axis)
    return a


def dec2d(img, wavelet, boundary, dec, level):
    rows_a, rows_d = dec(img, wavelet, boundary, level, axis=0)
    a, d1 = dec(rows_a, wavelet, boundary, level, axis=1)
    d2, d3 = dec(rows_d, wavelet, boundary, level, axis=1)
    return (a, d1, d2, d3)


def wavedec2d(img, wavelet, level, boundary="symm", dec=dwt, thread=None):
    a = img
    res = []
    for j in range(int(level)):
        if thread and not thread.is_alive():
            return None
        a, d1, d2, d3 = dec2d(a, wavelet, boundary, dec, j)
        res.append([d1, d2, d3])
    res.append(a)
    return res


def rec2d(a, d, wavelet, boundary, rec, level):
    d1, d2, d3 = d

    if a.shape != d1.shape:
        a = nputils.index(a, np.s_[:-1])
    temp_a = rec(a, d1, wavelet, boundary, level, axis=1)
    temp_d = rec(d2, d3, wavelet, boundary, level, axis=1)
    img = rec(temp_a, temp_d, wavelet, boundary, level, axis=0)
    return img


def waverec2d(coefs, wavelet, boundary="symm", rec=dwt_inv, shape=None, thread=None):
    a = coefs[-1]
    for j in range(len(coefs) - 2, -1, -1):
        if thread and not thread.is_alive():
            return None
        a = rec2d(a, coefs[j], wavelet, boundary, rec, j)
    if shape and shape != a.shape:
        a = nputils.index(a, np.s_[:-1])
    return a


def dyadic_image(coeffs, shape=None, normalize=True):

    def normalize(a):
        return (a - a.min()) / float(a.max() - a.min())

    if shape:
        d = [nputils.resize(coeffs[0], [k / pow(2., len(coeffs) - 1)
                                        for k in shape])]
        for l in range(1, len(coeffs)):
            s = [k / pow(2., len(coeffs) - l) for k in shape]
            d.append(map(nputils.resize, coeffs[l], [s] * 3))
    else:
        shape = coeffs[-1].shape
        d = coeffs
        for coef in coeffs[0:-1]:
            shape = shape + np.array(coef[0].shape)

    if normalize:
        # normalize aproximation
        d[-1] = normalize(d[-1])
        # normalize details
        for l in range(0, len(coeffs) - 1):
            d[l] = map(normalize, d[l])

    res = np.ones(shape)

    nputils.fill_at(res, (0, 0), d[-1])
    (x, y) = d[-1].shape
    for l in range(len(d) - 2, -1, -1):
        nputils.fill_at(res, (0, y), d[l][0])
        nputils.fill_at(res, (x, 0), d[l][1])
        nputils.fill_at(res, (x, y), d[l][2])
        (x, y) = d[l][0].shape + np.array((x, y))
    return res


def get_noise_factor_from_background(wavelet, level, dec, background):
    scales = wavedec(background, wavelet, level, dec=dec)
    return [scale.std() for scale in scales[:-1]]


def get_noise_factor_from_data(wavelet, level, dec, data):
    scales = wavedec(data, wavelet, level, dec=dec)
    return [nputils.k_sigma_noise_estimation(scale) for scale in scales[:-1]]


def get_noise_factor(wavelet, level, dec, beam=None):
    # n = (250000)
    n = (200, 200)
    background = nputils.gaussian_noise(n, 0, 1)
    if beam is not None:
        background = beam.convolve(background)
    return get_noise_factor_from_background(wavelet, level, dec, background)


def wave_noise_factor(bg, wavelet, level, dec, beam=None):
    if isinstance(bg, np.ndarray):
        scales_noise = get_noise_factor_from_background(wavelet, level, dec, bg)
    else:
        scales_noise = bg * np.array(get_noise_factor(wavelet, level, dec, beam=beam))
    return scales_noise


def dec_noise_factor(dec, bg, beam=None, **kargs):
    if not isinstance(bg, np.ndarray):
        n = (200, 200)
        bg = nputils.gaussian_noise(n, 0, bg)
        if beam is not None:
            bg = beam.convolve(bg)
    scales = dec(bg, **kargs)
    return [scale.std() for scale in scales[:-1]]


def dog_noise_factor(bg, widths=None, angle=0, ellipticity=1, beam=None):
    return dec_noise_factor(dogdec, bg, beam=beam, widths=widths, angle=angle, ellipticity=ellipticity)
