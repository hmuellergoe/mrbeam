import numpy as np

from imagingbase.ehtim_wrapper import EhtimWrapper, EhtimObsdata

from ehtim.imaging.pol_imager_utils import mcv_r, mcv, polchisqdata, polchisq, polchisqgrad, polregularizer, polregularizergrad, unpack_poltuple, embed_pol, make_q_image, make_u_image, make_p_image
import ehtim.image as image

##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = False #ANDREW TODO change this default in the future

MAXLS = 100 # maximum number of line searches in L-BFGS-B
NHIST = 100 # number of steps to store for hessian approx
MAXIT = 100 # maximum number of iterations
STOP = 1.e-100 # convergence criterion

DATATERMS = ['pvis','m','pbs']
REGULARIZERS = ['msimple', 'hw', 'ptv']

nit = 0 # global variable to track the iteration number in the plotting callback

class EhtimWrapperPol(EhtimWrapper):
    def __init__(self, Obsdata, InitIm, Prior, flux, d='vis', stokesv=False, amp=False, **kwargs):
        super().__init__(Obsdata, InitIm, Prior, flux, d='vis', **kwargs)
        
        #For pol, rescaling only addresses mvec, not Stokes I
        self.Prior.imvec *= self.rescaling
        self.InitIm.imvec *= self.rescaling
        self.flux *= self.rescaling
        
        self.d = d

        self.dataterm = self.d in DATATERMS
        self.regterm = self.d in REGULARIZERS
        assert self.dataterm or self.regterm
        
        # some kwarg default values
        self.pol_trans = kwargs.get('pol_trans', True)
        self.pol_solve = kwargs.get('pol_solve', (0,1,1))
        
        # Make sure data and regularizer options are ok
        #if (self.pol_prim!="amp_phase"):
        #    raise Exception("Only amp_phase pol_prim currently supported!")
        if (len(self.pol_solve)!=3):
            raise Exception("pol_solve tuple must have 3 entries!")

        # Catch scale and dimension problems
        imsize = np.max([Prior.xdim, Prior.ydim]) * Prior.psize
        uvmax = 1.0/Prior.psize
        uvmin = 1.0/imsize
        uvdists = Obsdata.unpack('uvdist')['uvdist']
        maxbl = np.max(uvdists)
        minbl = np.max(uvdists[uvdists > 0])
        maxamp = np.max(np.abs(Obsdata.unpack('amp')['amp']))
        
        
        if uvmax < maxbl:
            print("Warning! Pixel Spacing is larger than smallest spatial wavelength!")
        if uvmin > minbl:
            print("Warning! Field of View is smaller than largest nonzero spatial wavelength!")

        # convert polrep to stokes
        self.Prior = self.Prior.switch_polrep(polrep_out='stokes', pol_prim_out='I')
        self.InitIm = self.InitIm.switch_polrep(polrep_out='stokes', pol_prim_out='I')

        # Define embedding mask
        self.embed_mask = self.Prior.imvec > self.clipfloor
        
        
        # initial Stokes I image
        self.iimage = self.InitIm.imvec[self.embed_mask]
        self.nimage = len(self.iimage)
    
        # initial pol image
        if self.pol_trans:
            if len(self.InitIm.qvec) and (np.any(self.InitIm.qvec!=0) or np.any(self.InitIm.uvec!=0)):
                init1 = (np.abs(self.InitIm.qvec + 1j*self.InitIm.uvec) / self.InitIm.imvec)[self.embed_mask]
                init2 = (np.arctan2(self.InitIm.uvec, self.InitIm.qvec) / 2.0)[self.embed_mask]
            else:
                # !AC TODO get the actual zero baseline pol. frac from the data!??
                print("No polarimetric image in the initial image!")
                init1 = 0.2 * (np.ones(len(self.iimage)) + 1e-2 * np.random.rand(len(self.iimage)))
                init2 = np.zeros(len(self.iimage)) + 1e-2 * np.random.rand(len(self.iimage))
    
                # Change of variables    
            self.inittuple = np.array((self.iimage, init1, init2))
            self.xtuple =  mcv_r(self.inittuple)
    
        else:
            if len(self.InitIm.qvec) and (np.any(self.InitIm.qvec!=0) or np.any(self.InitIm.uvec!=0)):
                init1 = self.InitIm.qvec
                init2 = self.InitIm.uvec
            else:
                # !AC TODO get the actual zero baseline pol. frac from the data!??
                print("No polarimetric image in the initial image!")
                init1 = 0.2 * self.iimage
                init2 = 0 * self.iimage
                
            # Change of variables    
            self.inittuple = np.array((self.iimage, init1, init2))
            self.xtuple =  self.inittuple.copy()
            
        self.vdata = self.Obsdata.unpack(['vvis'], conj=True)['vvis']
        self.vsigma = self.Obsdata.unpack(['vsigma'], conj=True)['vsigma']
        
        self.mdata = self.Obsdata.unpack(['mamp'], conj=True)['mamp']
        self.msigma = self.Obsdata.unpack(['msigma'], conj=True)['msigma']
            
            
    # Get data and fourier matrices for the data terms
        if self.rml:       
            (self.data, self.sigma, self.A) = polchisqdata(self.Obsdata, self.Prior, self.embed_mask, self.d, **kwargs)
            if self.ttype == 'direct' or self.ttype == 'nfft':
                try:
                    self.A *= self.rescaling
                except:
                    self.A = list(self.A)
                    for i in range(len(self.A)):
                        self.A[i] *= self.rescaling
                    self.A = tuple(self.A)
            
        self.nit = 0
        
        if stokesv:
            self._chisq = self._vchisq
            self._chisqgrad = self._vchisqgrad
        elif amp:
            self._chisq = self._mchisq
            self._chisqgrad = self._mchisqgrad
        else:
            self._chisq = self._pchisq
            self._chisqgrad = self._pchisqgrad
               
    # Define the chi^2 and chi^2 gradient
    def _pchisq(self, imtuple):
        toret = polchisq(imtuple, self.A, self.data, self.sigma, self.d, ttype=self.ttype, mask=self.embed_mask, pol_trans=self.pol_trans)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret

    def _pchisqgrad(self, imtuple):
        if self.pol_trans:
            c = polchisqgrad(imtuple, self.A, self.data, self.sigma, self.d, ttype=self.ttype, mask=self.embed_mask, pol_trans=self.pol_trans)
            toret = c.reshape((3, self.InitIm.xdim*self.InitIm.ydim))
            if self.ttype == 'fast':
                toret *= self.rescaling
            return toret
        else:
            iimage = imtuple[0]
            pimage = make_p_image(imtuple, self.pol_trans)
            psamples = np.dot(self.A, pimage)
            pdiff = (self.data - psamples) / (self.sigma**2)
            zeros =  np.zeros(len(iimage))
            
            qimage = imtuple[1]
            uimage = imtuple[2]
            
            gradi = zeros

            if self.pol_solve[1]!=0:
                gradq = -np.real(np.dot(self.A.conj().T, pdiff)) / len(self.data)
            else:
                gradq = zeros

            if self.pol_solve[2]!=0:
                gradu = -np.imag(np.dot(self.A.conj().T, pdiff)) / len(self.data)
            else:
                gradu = zeros

            gradout = np.array((gradi, gradq, gradu))
            toret = gradout.reshape((3, self.InitIm.xdim*self.InitIm.ydim))
            if self.ttype == 'fast':
                toret *= self.rescaling
            return toret

        # Define the regularizer and regularizer gradient
    def _reg(self, imtuple):
        imtup = imtuple.copy()
        imtup[1] *= self.rescaling
        toret = polregularizer(imtup, self.embed_mask, self.flux, 
                              self.Prior.xdim, self.Prior.ydim, self.Prior.psize, self.d, **self.kwargs)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret
    
    def _reggrad(self, imtuple):
        imtup = imtuple.copy()
        imtup[1] *= self.rescaling
        c = polregularizergrad(imtup, self.embed_mask, self.flux, 
                              self.Prior.xdim, self.Prior.ydim, self.Prior.psize, self.d, **self.kwargs)
        toret = c.reshape((3, self.InitIm.xdim*self.InitIm.ydim))
        if self.ttype == 'fast':
            toret *= self.rescaling
        toret[1] /= self.rescaling
        return toret
    
    def formatoutput(self, res):
        out = res.flatten()
        out *= self.rescaling
        # Format output
        outcv = unpack_poltuple(out, self.xtuple, self.nimage, self.pol_solve)
        if self.pol_trans:
            outcut = mcv(outcv)  #change of variables
        else:
            outcut = outcv
    
        if np.any(np.invert(self.embed_mask)): 
            out = embed_pol(outcut, self.embed_mask) #embed
        else:
            out = outcut
    
        iimage = out[0]
        qimage = make_q_image(out, self.pol_trans)
        uimage = make_u_image(out, self.pol_trans)
    
        outim = image.Image(iimage.reshape(self.Prior.ydim, self.Prior.xdim), self.Prior.psize,
                             self.Prior.ra, self.Prior.dec, rf=self.Prior.rf, source=self.Prior.source,
                             mjd=self.Prior.mjd, pulse=self.Prior.pulse)
        outim.add_qu(qimage.reshape(self.Prior.ydim, self.Prior.xdim), uimage.reshape(self.Prior.ydim, self.Prior.xdim))
        
        return outim
    
    # Define the chi^2 and chi^2 gradient for Stokes V
    def _vchisq(self, vvec):
        
        if self.ttype != 'direct':
            raise NotImplementedError
        vsample = np.dot(self.A, vvec)
        chisq =  np.sum(np.abs((self.vdata - vsample))**2/(self.vsigma**2)) / (2*len(self.vdata)) 
        
        return chisq
    
    def _vchisqgrad(self, vvec):
        
        if self.ttype != 'direct':
            raise NotImplementedError
            
        vsample = np.dot(self.A, vvec)
        vdiff = (self.vdata - vsample) / (self.vsigma**2)
            
        gradv = -np.real(np.dot(self.A.conj().T, vdiff)) / len(self.vsigma)

        return gradv
    
    def _mchisq(self, imtuple):
        if self.ttype != 'direct':
            raise NotImplementedError
        iimage = imtuple[0]
        pimage = make_p_image(imtuple, self.pol_trans)
        msamples = np.dot(self.A, pimage) / np.dot(self.A, iimage)
        return np.sum((self.mdata-np.abs(msamples))**2/(self.msigma**2)) / (2*len(self.mdata)) 
    
    def _mchisqgrad(self, imtuple):
        assert self.pol_trans == False
        if self.ttype != 'direct':
            raise NotImplementedError
        iimage = imtuple[0]
        zeros =  np.zeros(len(iimage))
        pimage = make_p_image(imtuple, self.pol_trans)
        msamples = np.dot(self.A, pimage) / np.dot(self.A, iimage)
        mdiff = np.asarray((self.mdata-np.abs(msamples))/(self.msigma**2), dtype=complex)
        mdiff *= np.dot(self.A, pimage) / (np.abs(np.dot(self.A, iimage))*np.abs(np.dot(self.A, pimage)))
    
        qimage = imtuple[1]
        uimage = imtuple[2]
        
        gradi = zeros

        if self.pol_solve[1]!=0:
            gradq = -np.real(np.dot(self.A.conj().T, mdiff)) / len(self.mdata)
        else:
            gradq = zeros

        if self.pol_solve[2]!=0:
            gradu = -np.imag(np.dot(self.A.conj().T, mdiff)) / len(self.mdata)
        else:
            gradu = zeros

        gradout = np.array((gradi, gradq, gradu))
        
        return gradout
    
    def updateobs(self, obs):
        assert self.rml
        self.Obsdata = EhtimObsdata(obs, num_cores=self.num_cores)
                
        # Get data and fourier matrices for the data terms    
        (self.data, self.sigma, self.A) = polchisqdata(self.Obsdata, self.Prior, self.embed_mask, self.d, **self.kwargs)
        if self.ttype == 'direct' or self.ttype == 'nfft':
            try:
                self.A *= self.rescaling
            except:
                self.A = list(self.A)
                for i in range(len(self.A)):
                    self.A[i] *= self.rescaling
                self.A = tuple(self.A)