import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from imagingbase.regpy_functionals import Functional
from regpy.operators import Operator
from regpy.discrs import Discretization
from ehtim.imaging.imager_utils import chisqdata, chisq, chisqgrad, regularizer, regularizergrad, embed
from ehtim.obsdata import Obsdata
import ehtim.image as image
import ehtim.observing.obs_helpers as obsh
import ehtim.const_def as ehc

import ehtplot.color

from joblib import Parallel, delayed
from multiprocessing import Pool

##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = False

MAXLS = 100  # maximum number of line searches in L-BFGS-B
NHIST = 100  # number of steps to store for hessian approx
MAXIT = 100  # maximum number of iterations
STOP = 1.e-8  # convergence criterion

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag',
             'camp', 'logcamp', 'logcamp_diag', 'logamp']
REGULARIZERS = ['gs', 'tv', 'tv2', 'l1w', 'lA', 'patch', 'simple', 'compact', 'compact2', 'rgauss', 'flux']

nit = 0  # global variable to track the iteration number in the plotting callback

class EhtimWrapper():
    def __init__(self, Obsdata, InitIm, Prior, flux, d='vis', **kwargs):
        self.num_cores = kwargs.get('num_cores', 1)
        
        self.Obsdata = EhtimObsdata(Obsdata, num_cores=self.num_cores)
        self.InitIm = InitIm
        self.Prior = Prior
        self.flux = flux
        self.d = d

        self.dataterm = self.d in DATATERMS
        self.regterm = self.d in REGULARIZERS
        assert self.dataterm or self.regterm

        # some kwarg default values
        self.maxit = kwargs.get('maxit', MAXIT)
        self.stop = kwargs.get('stop', STOP)
        self.clipfloor = kwargs.get('clipfloor', 0)
        self.ttype = kwargs.get('ttype', 'direct')

        self.grads = kwargs.get('grads', True)
        self.logim = kwargs.get('logim', False)
        self.norm_init = kwargs.get('norm_init', False)
        self.show_updates = kwargs.get('show_updates', True)
        
        self.rml = kwargs.get('rml', True)

        self.rescaling = kwargs.get('rescaling', 1)
        self.Prior.imvec /= self.rescaling
        self.InitIm.imvec /= self.rescaling
        self.flux /= self.rescaling

        self.beam_size = kwargs.get('beam_size', Obsdata.res())

        self.kwargs = kwargs
        self.kwargs['beam_size'] = self.beam_size

        # Make sure data and regularizer options are ok
        if (self.Prior.psize != self.InitIm.psize) or (self.Prior.xdim != self.InitIm.xdim) or (self.Prior.ydim != self.InitIm.ydim):
            raise Exception("Initial image does not match dimensions of the prior image!")
        if (self.InitIm.polrep != self.Prior.polrep):
            raise Exception(
                "Initial image pol. representation does not match pol. representation of the prior image!")
        if (self.logim and self.Prior.pol_prim in ['Q', 'U', 'V']):
            raise Exception(
                "Cannot image Stokes Q,U,or V with log image transformation! Set logim=False in imager_func")

        self.pol = self.Prior.pol_prim
        print("Generating %s image..." % self.pol)

        # Catch scale and dimension problems
        imsize = np.max([self.Prior.xdim, self.Prior.ydim]) * self.Prior.psize
        uvmax = 1.0/self.Prior.psize
        uvmin = 1.0/imsize
        uvdists = Obsdata.unpack('uvdist')['uvdist']
        maxbl = np.max(uvdists)
        minbl = np.max(uvdists[uvdists > 0])
        maxamp = np.max(np.abs(self.Obsdata.unpack('amp')['amp']))

        if uvmax < maxbl:
            print("Warning! Pixel Spacing is larger than smallest spatial wavelength!")
        if uvmin > minbl:
            print("Warning! Field of View is smaller than largest nonzero spatial wavelength!")
        if flux > 1.2*maxamp:
            print("Warning! Specified flux is > 120% of maximum visibility amplitude!")
        if flux < .8*maxamp:
            print("Warning! Specified flux is < 80% of maximum visibility amplitude!")

        # Define embedding mask
        self.embed_mask = self.Prior.imvec > self.clipfloor

        # Normalize prior image to total flux and limit imager range to prior values > clipfloor
        if (not self.norm_init):
            self.nprior = self.Prior.imvec[self.embed_mask]
            ninit = self.InitIm.imvec[self.embed_mask]
        else:
            self.nprior = (self.flux * self.Prior.imvec / np.sum((self.Prior.imvec)[self.embed_mask]))[self.embed_mask]
            ninit = (self.flux * self.InitIm.imvec / np.sum((self.InitIm.imvec)[self.embed_mask]))[self.embed_mask]

        if len(self.nprior) == 0:
            raise Exception("clipfloor too large: all prior pixels have been clipped!")

        if self.rml:
            # Get data and fourier matrices for the data terms
            (self.data, self.sigma, self.A) = chisqdata(self.Obsdata, self.Prior, self.embed_mask, self.d, pol=self.pol, **kwargs)
            if self.ttype == 'direct' or self.ttype == 'nfft':
                try:
                    self.A *= self.rescaling
                except:
                    self.A = list(self.A)
                    for i in range(len(self.A)):
                        self.A[i] *= self.rescaling
                    self.A = tuple(self.A)

        self.nit = 0

    def _updatedatafidelityterm(self, d):
        assert self.rml
        self.d = d
        (self.data, self.sigma, self.A) = chisqdata(self.Obsdata, self.Prior, self.embed_mask, self.d, pol=self.pol, **self.kwargs)
        if self.ttype == 'direct' or self.ttype == 'nfft':
            try:
                self.A *= self.rescaling
            except:
                self.A = list(self.A)
                for i in range(len(self.A)):
                    self.A[i] *= self.rescaling
                self.A = tuple(self.A)
        return

        # Define the chi^2 and chi^2 gradient
    def _chisq(self, imvec):
        toret = chisq(imvec.flatten(), self.A, self.data, self.sigma, self.d, ttype=self.ttype, mask=self.embed_mask)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret

    def _chisqgrad(self, imvec):
        c = chisqgrad(imvec.flatten(), self.A, self.data, self.sigma, self.d, ttype=self.ttype, mask=self.embed_mask)
        toret = c.reshape(self.InitIm.xdim, self.InitIm.ydim)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret

        # Define the regularizer and regularizer gradient
    def _reg(self, imvec):
        toret = regularizer(imvec.flatten(), self.nprior, self.embed_mask, self.flux, self.Prior.xdim, self.Prior.ydim, self.Prior.psize, self.d, **self.kwargs)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret
    
    def _reggrad(self, imvec):
        c = regularizergrad(imvec.flatten(), self.nprior, self.embed_mask, self.flux, self.Prior.xdim, self.Prior.ydim, self.Prior.psize, self.d, **self.kwargs)
        toret = c.reshape(self.InitIm.xdim, self.InitIm.ydim)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret
    # Define plotting function for each iteration

    def _plotcur(self, im_step, **kwargs):
        if self.logim:
            im_step = np.exp(im_step)
        if self.show_updates:
            if np.any(np.invert(self.embed_mask)):
                im_step = embed(im_step, self.embed_mask)
            self._plot_i(im_step, self.Prior, self.nit, pol=self.pol, **kwargs)
        self.nit += 1

    def _plot_i(self, im, Prior, nit, **kwargs):
        cmap = kwargs.get('cmap', 'afmhot_u')
        interpolation = kwargs.get('interpolation', 'gaussian')
        pol = kwargs.get('pol', '')
        scale = kwargs.get('scale', None)
        dynamic_range = kwargs.get('dynamic_range', 1.e5)
        gamma = kwargs.get('dynamic_range', .5)
        vmax = kwargs.get('vmax', None)

        plt.ion()
        plt.pause(1.e-6)
        plt.clf()

        imarr = im.reshape(Prior.ydim, Prior.xdim).copy()
        
        imarr = imarr[Prior.ydim//4:3*Prior.ydim//4, Prior.xdim//4:3*Prior.xdim//4]

        if scale == 'log':
            if (imarr < 0.0).any():
                print('clipping values less than 0')
                imarr[imarr < 0.0] = 0.0
            imarr = np.log(imarr + np.max(imarr)/dynamic_range)

        if scale == 'gamma':
            if (imarr < 0.0).any():
                print('clipping values less than 0')
                imarr[imarr < 0.0] = 0.0
            imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)
            
        factor = 3.254e13 / (self.Obsdata.rf**2 * self.Prior.psize**2)
        imarr *= factor
        imarr *= self.rescaling

        plt.imshow(imarr, cmap=plt.get_cmap(cmap), interpolation=interpolation, vmin=0, vmax=vmax)
        xticks = obsh.ticks(Prior.xdim//2, Prior.psize/ehc.RADPERUAS)
        yticks = obsh.ticks(Prior.ydim//2, Prior.psize/ehc.RADPERUAS)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        plt.xlabel(r'Relative RA ($\mu$as)')
        plt.ylabel(r'Relative Dec ($\mu$as)')
        plotstr = str(pol) + " : step: %i  " % nit
        plt.title(plotstr, fontsize=18)

    def formatoutput(self, res):
        out = res.flatten()
        out *= self.rescaling
        # Format output
        if np.any(np.invert(self.embed_mask)):
            out = embed(res.flatten()*self.embed_mask, np.ones(self.embed_mask.shape))

        outim = image.Image(out.reshape(self.Prior.ydim, self.Prior.xdim),
                            self.Prior.psize, self.Prior.ra, self.Prior.dec,
                            rf=self.Prior.rf, source=self.Prior.source,
                            polrep=self.Prior.polrep, pol_prim=self.pol,
                            mjd=self.Prior.mjd, time=self.Prior.time, pulse=self.Prior.pulse)

        # copy over other polarizations
        #outim.copy_pol_images(self.InitIm)

        # Return Image object
        return outim

    def updateobs(self, obs):
        assert self.rml
        self.Obsdata = obs
        
        (self.data, self.sigma, self.A) = chisqdata(self.Obsdata, self.Prior, self.embed_mask, self.d, pol=self.pol, **self.kwargs)
        if self.ttype == 'direct' or self.ttype == 'nfft':
            try:
                self.A *= self.rescaling
            except:
                self.A = list(self.A)
                for i in range(len(self.A)):
                    self.A[i] *= self.rescaling
                self.A = tuple(self.A)

    def unpack(self, string, dtype='float', **kwargs):
        prop = self.Obsdata.unpack(string, **kwargs)
        return np.array(prop, dtype=dtype).flatten()
    
    def find_widths(self, threshold, fov=0.5, option='wide', ratio=0.1): 
        #load uv-distance
        uvdist = self.unpack('uvdist')

        uvdist = np.sort(uvdist)
        
        #find significant jumps
        jumps = np.zeros(len(uvdist)-1)
        
        for i in range(len(uvdist)-1):
            jumps[i] = uvdist[i+1]-uvdist[i]
                   
        indices = [jumps[i] > threshold for i in range(len(jumps))]
        
        indices.append(False)
        
        #find cuts to jumps        
        cuts_lower = uvdist[indices]
        cuts_upper = uvdist[1:][indices[:-1]]
        
        cuts = np.concatenate(( np.asarray([uvdist[0]]), cuts_lower, cuts_upper, np.asarray([uvdist[-1]]) ))
               
        cuts = np.sort(cuts)

        #sigma = 1/(2*np.pi*cuts) 
        
        #fwhm = 2.355 * sigma
        
        #widths = fwhm / self.Prior.psize
        
        #define sigmas from cuts, merge maximal width
        sigma = 1/cuts
        widths = sigma / self.Prior.psize
        widths = widths[::-1]
        
        widths_max = self.Prior.xdim * fov
        
        width_in_fov = np.zeros(len(widths), dtype=bool)
        for i in range(len(widths)):
            width_in_fov[i] = (widths[i] < widths_max)
        widths = widths[width_in_fov]
        
        if False in width_in_fov:
            widths = np.concatenate((widths, np.asarray([widths_max])))
        
        #widen field in gaps
        nr_gaps = (len(widths)-1)//2 
        delta = 1/cuts_lower[::-1] - 1/cuts_upper[::-1]
        delta = delta[0:nr_gaps] * ratio / self.Prior.psize
        
        if option == 'wide':
        
            for i in range(len(widths)-2):
                if i%2 == 0:
                    widths[1+i] = widths[1+i]+delta[i//2]
                else:
                    widths[1+i] = widths[1+i]-delta[i//2]
        
        if option == 'narrow':
            
            for i in range(len(widths)-2):
                if i%2 == 0:
                    widths[1+i] = widths[1+i]-delta[i//2]
                else:
                    widths[1+i] = widths[1+i]+delta[i//2]
        
        #add intermediate scales for large jumps in sigma        
        while True:
            widths_old = widths.copy()
            widths = self._update_widths(widths)
            if len(widths) == len(widths_old):
                break

        return widths
    
    def _update_widths(self, widths):
        for i in range(len(widths)-1):
            if widths[i+1]/widths[i] >= 3:
                to_add = np.asarray([widths[i]*2])
                widths = np.concatenate((to_add, widths))
                widths = np.sort(widths)
                return widths
        return widths


class EhtimFunctional(Functional):
    def __init__(self, handler, domain):
        self.handler = handler
        super().__init__(domain)

    def _eval(self, imvec):
        if self.handler.dataterm:
            return self.handler._chisq(imvec)
        else:
            return self.handler._reg(imvec)

    def _gradient(self, imvec):
        if self.handler.dataterm:
            return self.handler._chisqgrad(imvec)
        else:
            return self.handler._reggrad(imvec)

    def _proximal(self, imvec, tau):
        return NotImplementedError

class EhtimOperator(Operator):
    def __init__(self, handler, domain, codomain=None):
        assert handler.ttype == 'direct' or handler.ttype == 'nfft'
        self.handler = handler
        codomain = codomain or Discretization(self.handler.A.shape[0], dtype=complex)
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return self.handler.A @ x.flatten()

    def _adjoint(self, y):
        toret = np.conjugate(self.handler.A.T) @ y
        if self.domain.is_complex == False:
            toret = np.real(toret)
        return toret.reshape(self.domain.shape)

    
class EhtimObsdata(Obsdata):
    def __init__(self, obs, num_cores=1, gridding_correction=False):
        self.power = 0
        self.weight = 1
        self.num_cores = num_cores
        self.gridding_correction = gridding_correction
        self.update_obs = True
        super().__init__(obs.ra, obs.dec, obs.rf, obs.bw, obs.data, obs.tarr, scantable=obs.scans,
                 polrep=obs.polrep, source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
                 ampcal=obs.ampcal, phasecal=obs.phasecal, opacitycal=obs.opacitycal, dcal=obs, frcal=obs.frcal)

    def cleanbeam(self, npix, fov, pulse=ehc.PULSE_DEFAULT, flux=None):
        im = image.make_square(self, npix, fov, pulse=pulse)
        beamparams = self._fit_beam(npix, fov, flux=flux)
        im = im.add_gauss(1.0, beamparams)

        return im

    def _fit_beam(self, npix, fov, units='rad', flux=None):
        pdim = fov / npix
        u_org = self.unpack('u', conj=True)['u']
        v_org = self.unpack('v', conj=True)['v']
        sigma_org = self.unpack('sigma', conj=True)['sigma']
        
        self.u = np.zeros(len(u_org)+1)
        self.v = np.zeros(len(v_org)+1)
        sigma = np.zeros(len(sigma_org)+1)
        
        self.u[:-1] = u_org
        self.v[:-1] = v_org
        sigma[:-1] = sigma_org
        
        flux_im = 0
        sigma_flux = 0.1
        if flux == None:
            uvdist = self.unpack('uvdist', conj=True)['uvdist']
            index = np.argmin(uvdist)
            
            amp = self.unpack('amp', conj=True)['amp']
            flux_im += amp[index]
            sigma_flux = sigma[index]
        else:
            flux_im += flux
            
        sigma[-1] = sigma_flux

        self.weights = 1. / sigma**self.power
        
        gridu = np.linspace(-0.5/pdim, 0.5/pdim*npix/(npix+1), npix)
        gridv = np.linspace(-0.5/pdim, 0.5/pdim*npix/(npix+1), npix)
        uu, vv = np.asarray(np.meshgrid(gridu, gridv))
        self.grid = np.zeros((npix, npix, 2))
        self.grid[:,:,0] = uu
        self.grid[:,:,1] = vv
        gridded_mask = np.zeros((npix, npix))
        gridded_cells = np.zeros((npix, npix))
        
        pooling = Pool(self.num_cores)
        
        n = 100
        r = range(0, len(self.u), n)
        results = []
        
        self._fov = fov
        self._npix = npix
        
        for arg in zip([x+1 for x in r],r[1:]):
            results.append(pooling.apply_async(self._sum, arg))
            
        results.append(pooling.apply_async(self._sum, (r[-1], len(self.u)-1)))
            
        for i in range(len(results)):
            gridded_cells += results[i].get()[0]
            gridded_mask += results[i].get()[1]
            
        self.uniform_rescaling = np.maximum(gridded_cells, 1/(2*np.pi))**self.weight
        
        gridded_mask /= self.uniform_rescaling

        abc = np.array([np.sum(gridded_mask * uu**2),
                        np.sum(gridded_mask * vv**2),
                        2 * np.sum(gridded_mask * uu * vv)])
        abc *= (2. * np.pi**2 / np.sum(gridded_mask))
        abc *= 1e-20  # Decrease size of coefficients

        # Fit the beam
        guess = [(50)**2, (50)**2, 0.0]
        params = opt.minimize(self._fit_chisq, guess, args=(abc,), method='Powell')

        # Return parameters, adjusting fwhm_maj and fwhm_min if necessary
        if params.x[0] > params.x[1]:
            fwhm_maj = 1e-10 * np.sqrt(params.x[0])
            fwhm_min = 1e-10 * np.sqrt(params.x[1])
            theta = np.mod(params.x[2], np.pi)
        else:
            fwhm_maj = 1e-10 * np.sqrt(params.x[1])
            fwhm_min = 1e-10 * np.sqrt(params.x[0])
            theta = np.mod(params.x[2] + np.pi / 2.0, np.pi)

        gparams = np.array((fwhm_maj, fwhm_min, theta))

        if units == 'natural':
            gparams[0] /= ehc.RADPERUAS
            gparams[1] /= ehc.RADPERUAS
            gparams[2] *= 180. / np.pi

        return gparams

    def _fit_chisq(self, beamparams, db_coeff):

        (fwhm_maj2, fwhm_min2, theta) = beamparams
        a = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_min2 + np.sin(theta)**2 / fwhm_maj2)
        b = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_maj2 + np.sin(theta)**2 / fwhm_min2)
        c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1.0 / fwhm_maj2 - 1.0 / fwhm_min2)
        gauss_coeff = np.array((a, b, c))

        chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)

        return chisq 
            
    def uvweight(self, power, weight=1):
        self.power = power
        self.weight = weight
        self.update_obs = True
        
    def dirtybeam(self, npix, fov, pulse=ehc.PULSE_DEFAULT, flux=None, natural_correction=False):
        dmap, dbeam = self.dirty_image_beam(npix, fov, pulse=pulse, flux=flux, natural_correction=natural_correction)
        return dbeam
    
    def dirtyimage(self, npix, fov, pulse=ehc.PULSE_DEFAULT, flux=None, natural_correction=False):
        dmap, dbeam = self.dirty_image_beam(npix, fov, pulse=pulse, flux=flux, natural_correction=natural_correction)
        return dmap
        
    def dirty_image_beam(self, npix, fov, pulse=ehc.PULSE_DEFAULT, flux=None, natural_correction=False):       
        pdim = fov / npix
        u_org = self.unpack('u', conj=True)['u']
        v_org = self.unpack('v', conj=True)['v']
        sigma_org = self.unpack('sigma', conj=True)['sigma']
        
        self.u = np.zeros(len(u_org)+1)
        self.v = np.zeros(len(v_org)+1)
        sigma = np.zeros(len(sigma_org)+1)
        
        self.u[:-1] = u_org
        self.v[:-1] = v_org
        sigma[:-1] = sigma_org
        
        flux_im = 0
        sigma_flux = 0.1
        if flux == None:
            uvdist = self.unpack('uvdist', conj=True)['uvdist']
            index = np.argmin(uvdist)
            
            amp = self.unpack('amp', conj=True)['amp']
            flux_im += amp[index]
            sigma_flux = sigma[index]
        else:
            flux_im += flux
            
        sigma[-1] = sigma_flux
        
        visname = self.poldict["vis1"]

        vis_org = self.unpack(visname, conj=True)[visname]
        self.vis = np.zeros(len(vis_org)+1, dtype=complex)
        self.vis[:-1] = vis_org
        self.vis[-1] = flux_im

        self.weights = 1. / sigma**self.power
        
        if self.update_obs:
            gridu = np.linspace(-0.5/pdim, 0.5/pdim*npix/(npix+1), npix)
            gridv = np.linspace(-0.5/pdim, 0.5/pdim*npix/(npix+1), npix)
            uu, vv = np.asarray(np.meshgrid(gridu, gridv))
            self.grid = np.zeros((npix, npix, 2))
            self.grid[:,:,0] = uu
            self.grid[:,:,1] = vv
            self.gridded_cells_squared = np.zeros((npix, npix))
            self.gridded_mask = np.zeros((npix, npix))
            self.gridded_cells = np.zeros((npix, npix))
            self.gridded_vis = np.zeros((npix, npix), dtype=complex)
            
            pooling = Pool(self.num_cores)
            
            n = 100
            r = range(0, len(self.u), n)
            results = []
            
            self._fov = fov
            self._npix = npix
            
            for arg in zip([x+1 for x in r],r[1:]):
                results.append(pooling.apply_async(self._sum, arg))
                
            results.append(pooling.apply_async(self._sum, (r[-1], len(self.u)-1)))
                
            for i in range(len(results)):
                self.gridded_cells += results[i].get()[0]
                self.gridded_cells_squared += results[i].get()[1]
                self.gridded_mask += results[i].get()[2]
                self.gridded_vis += results[i].get()[3]
            
            #results = Parallel(n_jobs=self.num_cores)(delayed(self._gridding)(grid, u[i], v[i], fov) for i in range(len(u)))
            #for i in range(len(u)):
            #    gridded_cells += results[i].reshape(gridded_mask.shape)
            #    gridded_mask += weights[i] * results[i].reshape(gridded_mask.shape)
            #    gridded_vis += vis[i] * weights[i] * results[i].reshape(gridded_vis.shape)
                            
            self.uniform_rescaling = np.maximum(self.gridded_cells, 1/(2*np.pi))**self.weight
            self.natural_rescaling = np.maximum(np.sqrt(self.gridded_cells_squared), 1/(2*np.pi))
            
        else:
            if fov != self.fov or npix != self.npix:
                gridu = np.linspace(-0.5/pdim, 0.5/pdim*npix/(npix+1), npix)
                gridv = np.linspace(-0.5/pdim, 0.5/pdim*npix/(npix+1), npix)
                uu, vv = np.asarray(np.meshgrid(gridu, gridv))
                self.grid = np.zeros((npix, npix, 2))
                self.grid[:,:,0] = uu
                self.grid[:,:,1] = vv
                self.gridded_cells_squared = np.zeros((npix, npix))
                self.gridded_mask = np.zeros((npix, npix))
                self.gridded_cells = np.zeros((npix, npix))
                self.gridded_vis = np.zeros((npix, npix), dtype=complex)
                
                pooling = Pool(self.num_cores)
                
                n = 100
                r = range(0, len(self.u), n)
                results = []
                
                self._fov = fov
                self._npix = npix
                
                for arg in zip([x+1 for x in r],r[1:]):
                    results.append(pooling.apply_async(self._sum, arg))
                    
                results.append(pooling.apply_async(self._sum, (r[-1], len(self.u)-1)))
                    
                for i in range(len(results)):
                    self.gridded_cells += results[i].get()[0]
                    self.gridded_cells_squared += results[i].get()[1]
                    self.gridded_mask += results[i].get()[2]
                    self.gridded_vis += results[i].get()[3]
                
                #results = Parallel(n_jobs=self.num_cores)(delayed(self._gridding)(grid, u[i], v[i], fov) for i in range(len(u)))
                #for i in range(len(u)):
                #    gridded_cells += results[i].reshape(gridded_mask.shape)
                #    gridded_mask += weights[i] * results[i].reshape(gridded_mask.shape)
                #    gridded_vis += vis[i] * weights[i] * results[i].reshape(gridded_vis.shape)
                                
                self.uniform_rescaling = np.maximum(self.gridded_cells, 1/(2*np.pi))**self.weight
                self.natural_rescaling = np.maximum(np.sqrt(self.gridded_cells_squared), 1/(2*np.pi))
                
        
        gridding_function = self._gridding(self.grid, 0, 0, fov)
        gridding_conv = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gridding_function))))
        peak = np.max(gridding_conv)
        indices = gridding_conv > 0.0001 * peak
        gridding_correction = np.zeros(gridding_conv.shape)
        gridding_correction[indices] = 1 / gridding_conv[indices]
        
        if self.gridding_correction == False:
            gridding_correction = np.ones(gridding_conv.shape)
                
        gridded_mask = self.gridded_mask/self.uniform_rescaling
        if natural_correction:
            gridded_mask /= self.natural_rescaling
        
        dim = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gridded_mask))))
        dim *= gridding_correction
        
        normfac = 1 / np.sum(dim) #weights[-1] / np.sum(dim)
        dim *= normfac
            
        gridded_vis = self.gridded_vis/self.uniform_rescaling   
        if natural_correction:
            gridded_vis /= self.natural_rescaling
            
        im = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gridded_vis))))
        im *= gridding_correction

        # Final normalization
        im = im * normfac
        im = im[0:npix, 0:npix]


        dmap = image.Image(im, pdim, self.ra, self.dec, polrep=self.polrep,
                                rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
        
        dbeam = image.Image(dim, pdim, self.ra, self.dec, polrep=self.polrep,
                                rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
        
        self.update_obs = False
        self.fov = fov
        self.npix = npix

        return [dmap, dbeam]
    
    def _sum(self, low, high):
        result_cells = np.zeros((self._npix, self._npix))
        result_cells_squared = np.zeros((self._npix, self._npix))
        result_mask = np.zeros((self._npix, self._npix))
        result_vis = np.zeros((self._npix, self._npix), dtype=complex)
        for i in range(low, high+1):
            result = self._gridding(self.grid, self.u[i], self.v[i], self._fov)
            result_cells += result
            result_cells_squared += result**2
            result_mask += self.weights[i] * result
            result_vis += self.vis[i] * self.weights[i] * result
        return [result_cells, result_cells_squared, result_mask, result_vis]
    
    def _gridding(self, grid, u, v, fov):
        diff = np.linalg.norm(grid-np.asarray([u, v]), axis=2)
        toadd = np.zeros(diff.shape)
        indices = (diff < 10/fov)
        toadd[indices] = self._gaussian(diff[indices],1/fov)
        return toadd

    def _gaussian(self, r, sigma):
        return 1/(2*np.pi)*np.exp(-0.5*r**2/sigma**2)      
    
    
    def flag_nan(self, field, output='kept'):
        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()
        
        totest = self.unpack([field])[field]
        
        mask = np.isnan(totest)
        
        datatable_flagged = datatable_kept[mask]
        datatable_kept = datatable_kept[np.invert(mask)]
            
        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept    
            
            
            
            
            
            