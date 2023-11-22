import numpy as np
import resolve as rve
import nifty8 as ift

import ehtim as eh
from ehtim.imaging.imager_utils import REGULARIZERS, DATATERMS
import ehtim.observing.obs_helpers as obsh
import ehtim.const_def as ehc

from imagingbase.ehtim_wrapper import EhtimWrapper

from regpy.operators import Operator
from regpy.discrs import Discretization

class Response(Operator):
    def __init__(self, Obsdata, InitIm, do_wgridding, epsilon, useuvw=False, uvw=None):
        times = Obsdata.data["time"]
        antennas = {}
        for i in range(len(Obsdata.tarr)):
            antennas[Obsdata.tarr[i][0]] = int(i)
        t1 = Obsdata.data["t1"]
        t2 = Obsdata.data["t2"]
        ant1 = np.zeros(t1.shape, dtype=int)
        ant2 = np.zeros(t2.shape, dtype=int)
        for i in range(len(t1)):
            ant1[i] = antennas[t1[i]]
            ant2[i] = antennas[t2[i]]
        if useuvw:
            antenna_positions = rve.AntennaPositions(uvw, ant1=ant1, ant2=ant2, time=times)
        else:
            u = Obsdata.data["u"]
            v = Obsdata.data["v"]
            w = np.zeros(u.shape)
            uvw = np.hstack((u.reshape(-1,1), v.reshape(-1,1), w.reshape(-1,1)))
            antenna_positions = rve.AntennaPositions(uvw, ant1=ant1, ant2=ant2, time=times)
        
        vis = Obsdata.data['vis']
        vis = vis.reshape((1, len(vis), 1))
        
        weight = 1/Obsdata.data['sigma']
        weight = weight.reshape((1, len(weight), 1))
        
        polarization = rve.Polarization(())
        
        mock_observation = rve.Observation(antenna_positions, vis, weight, polarization, np.array([Obsdata.rf]))
        
        dx = InitIm.fovx() / (InitIm.xdim)
        dy = InitIm.fovy() / (InitIm.ydim)        
        space = ift.RGSpace((InitIm.xdim, InitIm.ydim), (dx, dy))
        
        measurement_domain = (rve.PolarizationSpace('I'),) + (rve.IRGSpace(np.zeros(1)),)*2 + (space,)
        pre_response = ift.DomainChangerAndReshaper(space,measurement_domain)
        
        # Set up response
        R = rve.InterferometryResponse(mock_observation, pre_response.target, do_wgridding=do_wgridding, epsilon=epsilon)
        self.R = R @ pre_response
        
        domain = Discretization(self.R.domain.shape[0]*self.R.domain.shape[1])
        codomain = Discretization(self.R.target.shape[1], dtype=complex)
        
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        inp = ift.makeField(self.R.domain, x.reshape(self.R.domain.shape))
        return self.R(inp).val.flatten()
        
    def _adjoint(self, y):
        inp = ift.makeField(self.R.target, y.reshape(self.R.target.shape))
        return self.R.adjoint(inp).val.flatten()

class ResolveWrapper(EhtimWrapper):
    def __init__(self, Obsdata, InitIm, Prior, flux, d='vis', **kwargs):
        super().__init__(Obsdata, InitIm, Prior, flux, d='vis', **kwargs)
        
        if self.d == 'vis' or self.d == 'amp' or self.d in REGULARIZERS:
            self.R = self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6)
        if self.d == 'bs':
            count = 'min'
            snrcut = 0
            weighting = 'natural'
            pol = 'I'

            # unpack data
            vtype = ehc.vis_poldict[pol]
            if (Obsdata.bispec is None) or (len(Obsdata.bispec) == 0) or pol != 'I':
                biarr = Obsdata.bispectra(mode="all", vtype=vtype, count=count, snrcut=snrcut)

            else:  # TODO -- pre-computed  with not stokes I?
                print("Using pre-computed bispectrum table in cphase chi^2!")
                if not type(Obsdata.bispec) in [np.ndarray, np.recarray]:
                    raise Exception("pre-computed bispectrum table is not a numpy rec array!")
                biarr = Obsdata.bispec
                # reduce to a minimal set
                if count != 'max':
                    biarr = obsh.reduce_tri_minimal(Obsdata, biarr)

            uv1 = np.hstack((biarr['u1'].reshape(-1, 1), biarr['v1'].reshape(-1, 1), np.zeros(len(biarr)).reshape(-1,1)))
            uv2 = np.hstack((biarr['u2'].reshape(-1, 1), biarr['v2'].reshape(-1, 1), np.zeros(len(biarr)).reshape(-1,1)))
            uv3 = np.hstack((biarr['u3'].reshape(-1, 1), biarr['v3'].reshape(-1, 1), np.zeros(len(biarr)).reshape(-1,1)))
            self.R = [self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv1),
                      self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv2),
                      self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv3)]
        if self.d == 'cphase':
            uv_min = False
            count = 'min'
            snrcut = 0
            systematic_cphase_noise = 0
            weighting = 'natural'

            # unpack data
            vtype = ehc.vis_poldict['I']
            if (Obsdata.cphase is None) or (len(Obsdata.cphase) == 0) or pol != 'I':
                clphasearr = Obsdata.c_phases(mode="all", vtype=vtype,
                                              count=count, uv_min=uv_min, snrcut=snrcut)
            else:  # TODO precomputed with not Stokes I
                print("Using pre-computed cphase table in cphase chi^2!")
                if not type(Obsdata.cphase) in [np.ndarray, np.recarray]:
                    raise Exception("pre-computed closure phase table is not a numpy rec array!")
                clphasearr = Obsdata.cphase
                # reduce to a minimal set
                if count != 'max':
                    clphasearr = obsh.reduce_tri_minimal(Obsdata, clphasearr)

            uv1 = np.hstack((clphasearr['u1'].reshape(-1, 1), clphasearr['v1'].reshape(-1, 1), np.zeros(len(clphasearr)).reshape(-1,1)))
            uv2 = np.hstack((clphasearr['u2'].reshape(-1, 1), clphasearr['v2'].reshape(-1, 1), np.zeros(len(clphasearr)).reshape(-1,1)))
            uv3 = np.hstack((clphasearr['u3'].reshape(-1, 1), clphasearr['v3'].reshape(-1, 1), np.zeros(len(clphasearr)).reshape(-1,1)))
            self.R = [self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv1),
                      self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv2),
                      self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv3)]
        if self.d == 'logcamp' or self.d == 'camp':
            uv_min = False
            count = 'min'
            snrcut = 0
            systematic_cphase_noise = 0
            weighting = 'natural'

            # unpack data & mask low snr points
            vtype = ehc.vis_poldict['I']
            if (Obsdata.logcamp is None) or (len(Obsdata.logcamp) == 0) or pol != 'I':
                clamparr = Obsdata.c_amplitudes(mode='all', count=count,
                                                vtype=vtype, ctype='logcamp', debias=False, snrcut=snrcut)
            else:  # TODO -- pre-computed  with not stokes I?
                print("Using pre-computed log closure amplitude table in log closure amplitude chi^2!")
                if not type(Obsdata.logcamp) in [np.ndarray, np.recarray]:
                    raise Exception("pre-computed log closure amplitude table is not a numpy rec array!")
                clamparr = Obsdata.logcamp
                # reduce to a minimal set
                if count != 'max':
                    clamparr = obsh.reduce_quad_minimal(Obsdata, clamparr, ctype='logcamp')
            uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1), np.zeros(len(clamparr)).reshape(-1,1)))
            uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1), np.zeros(len(clamparr)).reshape(-1,1)))
            uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1), np.zeros(len(clamparr)).reshape(-1,1)))
            uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1), np.zeros(len(clamparr)).reshape(-1,1)))
            self.R = [self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv1),
                      self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv2),
                      self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv3),
                      self.rescaling * Response(Obsdata, InitIm, do_wgridding=False, epsilon=1e-6, useuvw=True, uvw=uv3)]

        # Define the chi^2 and chi^2 gradient
    def _chisq(self, imvec):
        toret = chisq(imvec.flatten(), self.R, self.data, self.sigma, self.d, mask=self.embed_mask)
        return toret

    def _chisqgrad(self, imvec):
        c = chisqgrad(imvec.flatten(), self.R, self.data, self.sigma, self.d, mask=self.embed_mask)
        toret = c.reshape(self.InitIm.xdim, self.InitIm.ydim)
        return toret
 ##################################################################################################
 # Wrapper Functions
 ##################################################################################################


def chisq(imvec, A, data, sigma, dtype, mask=None):
     """return the chi^2 for the appropriate dtype
     """

     if mask is None:
         mask = []
     chisq = 1
     if dtype not in DATATERMS:
         return chisq

     if dtype == 'vis':
         chisq = chisq_vis(imvec, A, data, sigma)
     elif dtype == 'amp':
         chisq = chisq_amp(imvec, A, data, sigma)
#         elif dtype == 'logamp':
#             chisq = chisq_logamp(imvec, A, data, sigma)
     elif dtype == 'bs':
         chisq = chisq_bs(imvec, A, data, sigma)
     elif dtype == 'cphase':
         chisq = chisq_cphase(imvec, A, data, sigma)
#         elif dtype == 'cphase_diag':
#             chisq = chisq_cphase_diag(imvec, A, data, sigma)
     elif dtype == 'camp':
         chisq = chisq_camp(imvec, A, data, sigma)
     elif dtype == 'logcamp':
         chisq = chisq_logcamp(imvec, A, data, sigma)
#         elif dtype == 'logcamp_diag':
#             chisq = chisq_logcamp_diag(imvec, A, data, sigma)

     return chisq


def chisqgrad(imvec, A, data, sigma, dtype, mask=None):
     """return the chi^2 gradient for the appropriate dtype
     """

     if mask is None:
         mask = []
     chisqgrad = np.zeros(len(imvec))
     if dtype not in DATATERMS:
         return chisqgrad


     if dtype == 'vis':
         chisqgrad = chisqgrad_vis(imvec, A, data, sigma)
     elif dtype == 'amp':
         chisqgrad = chisqgrad_amp(imvec, A, data, sigma)
#         elif dtype == 'logamp':
#             chisqgrad = chisqgrad_logamp(imvec, A, data, sigma)
     elif dtype == 'bs':
         chisqgrad = chisqgrad_bs(imvec, A, data, sigma)
     elif dtype == 'cphase':
         chisqgrad = chisqgrad_cphase(imvec, A, data, sigma)
#         elif dtype == 'cphase_diag':
#             chisqgrad = chisqgrad_cphase_diag(imvec, A, data, sigma)
     elif dtype == 'camp':
         chisqgrad = chisqgrad_camp(imvec, A, data, sigma)
     elif dtype == 'logcamp':
         chisqgrad = chisqgrad_logcamp(imvec, A, data, sigma)
#         elif dtype == 'logcamp_diag':
#             chisqgrad = chisqgrad_logcamp_diag(imvec, A, data, sigma)

     return chisqgrad

##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis(imvec, A, vis, sigma):
    """Visibility chi-squared"""

    samples = A(imvec)
    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))
    return chisq

def chisqgrad_vis(imvec, A, vis, sigma):
    """The gradient of the visibility chi-squared"""

    samples = A(imvec)
    wdiff = (vis - samples)/(sigma**2)

    out = -np.real(A.adjoint(wdiff))/len(vis)
    return out


def chisq_amp(imvec, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""

    amp_samples = np.abs(A(imvec))
    return np.sum(np.abs((amp - amp_samples)/sigma)**2)/len(amp)


def chisqgrad_amp(imvec, A, amp, sigma):
    """The gradient of the amplitude chi-squared"""

    i1 = A(imvec)
    amp_samples = np.abs(i1)

    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out


def chisq_bs(imvec, A, bis, sigma):
    """Bispectrum chi-squared"""

    bisamples = (A[0](imvec) *
                 A[1](imvec) *
                 A[2](imvec))
    chisq = np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    return chisq


def chisqgrad_bs(imvec, A, bis, sigma):
    """The gradient of the bispectrum chi-squared"""

    bisamples = (A[0](imvec) *
                 A[1](imvec) *
                 A[2](imvec))

    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * A[1](imvec) * A[2](imvec)
    pt2 = wdiff * A[0](imvec) * A[2](imvec)
    pt3 = wdiff * A[0](imvec) * A[1](imvec)
    out = (A[0].adjoint(pt1) +
           A[1].adjoint(pt2) + 
           A[2].adjoint(pt3))

    out = -np.real(out) / len(bis)
    return out


def chisq_cphase(imvec, A, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    i1 = A[0](imvec)
    i2 = A[1](imvec)
    i3 = A[2](imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq


def chisqgrad_cphase(imvec, A, clphase, sigma):
    """The gradient of the closure phase chi-squared"""
    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    i1 = A[0](imvec)
    i2 = A[1](imvec)
    i3 = A[2](imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    pref = np.sin(clphase - clphase_samples)/(sigma**2)
    pt1 = pref/i1
    pt2 = pref/i2
    pt3 = pref/i3
    out = A[0].adjoint(pt1) + A[1].adjoint(pt2) + A[2].adjoint(pt3)
    out = (-2.0/len(clphase)) * np.imag(out)
    return out


def chisq_camp(imvec, A, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""

    i1 = A[0](imvec)
    i2 = A[1](imvec)
    i3 = A[2](imvec)
    i4 = A[3](imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq


def chisqgrad_camp(imvec, A, clamp, sigma):
    """The gradient of the closure amplitude chi-squared"""

    i1 = A[0](imvec)
    i2 = A[1](imvec)
    i3 = A[2](imvec)
    i4 = A[3](imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 = pp/i1
    pt2 = pp/i2
    pt3 = -pp/i3
    pt4 = -pp/i4
    out = (A[0].adjoint(pt1) +
           A[1].adjoint(pt2) +
           A[2].adjoint(pt3) +
           A[3].adjoint(pt4))
    out *= (-2.0/len(clamp)) * np.real(out)
    return out


def chisq_logcamp(imvec, A, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared"""

    a1 = np.abs(A[0](imvec))
    a2 = np.abs(A[1](imvec))
    a3 = np.abs(A[2](imvec))
    a4 = np.abs(A[3](imvec))

    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))
    return chisq


def chisqgrad_logcamp(imvec, A, log_clamp, sigma):
    """The gradient of the Log closure amplitude chi-squared"""

    i1 = np.abs(A[0](imvec))
    i2 = np.abs(A[1](imvec))
    i3 = np.abs(A[2](imvec))
    i4 = np.abs(A[3](imvec))
    log_clamp_samples = (np.log(np.abs(i1)) +
                         np.log(np.abs(i2)) - 
                         np.log(np.abs(i3)) -
                         np.log(np.abs(i4)))

    pp = (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / i1
    pt2 = pp / i2
    pt3 = -pp / i3
    pt4 = -pp / i4
    out = (A[0].adjoint(pt1) +
           A[1].adjoint(pt2) +
           A[2].adjoint(pt3) +
           A[3].adjoint(pt4))
    out = (-2.0/len(log_clamp)) * np.real(out)
    return out

































