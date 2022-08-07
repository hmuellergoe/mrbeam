import numpy as np

from imagingbase.ehtim_wrapper import EhtimWrapper

from ehtim.imaging.pol_imager_utils import mcv_r, mcv, polchisqdata, polchisq, polregularizer, polregularizergrad, unpack_poltuple, embed_pol, make_q_image, make_u_image
import ehtim.image as image

##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = False

MAXLS = 100 # maximum number of line searches in L-BFGS-B
NHIST = 100 # number of steps to store for hessian approx
MAXIT = 100 # maximum number of iterations
STOP = 1.e-100 # convergence criterion

DATATERMS = ['pvis','m','pbs']
REGULARIZERS = ['msimple', 'hw', 'ptv']

nit = 0 # global variable to track the iteration number in the plotting callback

class EhtimWrapperPol(EhtimWrapper):
    def __init__(self, Obsdata, InitIm, Prior, flux, d='vis', **kwargs):
        super().__init__(Obsdata, InitIm, Prior, flux, d='vis', **kwargs)
        self.d = d

        self.dataterm = self.d in DATATERMS
        self.regterm = self.d in REGULARIZERS
        assert self.dataterm or self.regterm
        
        # some kwarg default values
        self.pol_prim = kwargs.get('pol_prim', 'amp_phase')
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
        if self.pol_prim ==  "amp_phase":
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
    
        elif self.pol_prim == "qu":
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
        
    # Define the chi^2 and chi^2 gradient
    def _chisq(self, imtuple):
        toret = polchisq(imtuple, self.A, self.data, self.sigma, self.d, ttype=self.ttype, mask=self.embed_mask, pol_prim=self.pol_prim)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret

    def _chisqgrad(self, imtuple):
        c = polchisqgrad(imtuple, self.A, self.data, self.sigma, self.d, ttype=self.ttype, mask=self.embed_mask, pol_prim=self.pol_prim)
        toret = c.reshape((3, self.InitIm.xdim*self.InitIm.ydim))
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret

        # Define the regularizer and regularizer gradient
    def _reg(self, imtuple):
        toret = polregularizer(imtuple, self.embed_mask, self.flux, 
                              self.Prior.xdim, self.Prior.ydim, self.Prior.psize, self.d, **self.kwargs)
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret
    
    def _reggrad(self, imtuple):
        c = polregularizergrad(imtuple, self.embed_mask, self.flux, 
                              self.Prior.xdim, self.Prior.ydim, self.Prior.psize, self.d, **self.kwargs)
        toret = c.reshape((3, self.InitIm.xdim*self.InitIm.ydim))
        if self.ttype == 'fast':
            toret *= self.rescaling
        return toret
    
    def formatoutput(self, res):
        out = res.flatten()
        out *= self.rescaling
        # Format output
        outcv = unpack_poltuple(out, self.xtuple, self.nimage, self.pol_solve)
        if self.pol_prim == "amp_phase":
            outcut = mcv(outcv)  #change of variables
        else:
            outcut = outcv
    
        if np.any(np.invert(self.embed_mask)): 
            out = embed_pol(outcut, self.embed_mask) #embed
        else:
            out = outcut
    
        iimage = out[0]
        qimage = make_q_image(out, self.pol_prim)
        uimage = make_u_image(out, self.pol_prim)
    
        outim = image.Image(iimage.reshape(self.Prior.ydim, self.Prior.xdim), self.Prior.psize,
                             self.Prior.ra, self.Prior.dec, rf=self.Prior.rf, source=self.Prior.source,
                             mjd=self.Prior.mjd, pulse=self.Prior.pulse)
        outim.add_qu(qimage.reshape(self.Prior.ydim, self.Prior.xdim), uimage.reshape(self.Prior.ydim, self.Prior.xdim))
        
        return outim
    
from ehtim.imaging.pol_imager_utils import chisqgrad_m, chisqgrad_pbs, chisqgrad_p_nfft, chisqgrad_m_nfft, chisqgrad_pbs, make_p_image   
    
def polchisqgrad(imtuple, A, data, sigma, dtype, ttype='direct',
                 mask=[], pol_prim="amp_phase",pol_solve=(0,1,1)):
    
    """return the chi^2 gradient for the appropriate dtype
    """

    chisqgrad = np.zeros((3,len(imtuple[0])))
    if not dtype in DATATERMS:
        return chisqgrad
    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast' and 'direct'!")

    if ttype == 'direct':
        if dtype == 'pvis':
            chisqgrad = chisqgrad_p(imtuple, A, data, sigma, pol_prim,pol_solve)

        elif dtype == 'm':
            chisqgrad = chisqgrad_m(imtuple, A, data, sigma, pol_prim,pol_solve)

        elif dtype == 'pbs':
            chisqgrad = chisqgrad_pbs(imtuple, A, data, sigma, pol_prim,pol_solve)
    
    elif ttype== 'fast':
        raise Exception("FFT not yet implemented in polchisqgrad!")

    elif ttype== 'nfft':
        if len(mask)>0 and np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)

        if dtype == 'pvis':
            chisqgrad = chisqgrad_p_nfft(imtuple, A, data, sigma, pol_prim,pol_solve)

        elif dtype == 'm':
            chisqgrad = chisqgrad_m_nfft(imtuple, A, data, sigma, pol_prim,pol_solve)

        elif dtype == 'pbs':
            chisqgrad = chisqgrad_pbs(imtuple, A, data, sigma, pol_prim,pol_solve)

        if len(mask)>0 and np.any(np.invert(mask)):
            chisqgrad = np.array((chisqgrad[0][mask],chisqgrad[1][mask],chisqgrad[2][mask]))

    return chisqgrad    
    
    
def chisqgrad_p(imtuple, Amatrix, p, sigmap, pol_prim="amp_phase",pol_solve=(0,1,1)):
    """Polarimetric ratio chi-squared gradient
    """
    

    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_prim)
    psamples = np.dot(Amatrix, pimage)
    pdiff = (p - psamples) / (sigmap**2)
    zeros =  np.zeros(len(iimage))
        
    if pol_prim=="amp_phase":

        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        if pol_solve[0]!=0:
            gradi = -np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradi = zeros

        if pol_solve[1]!=0:
            gradm = -np.real(iimage*np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradm = zeros

        if pol_solve[2]!=0:
            gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradchi = zeros

        gradout = np.array((gradi, gradm, gradchi))
        
    elif pol_prim=="qu":
        
        qimage = imtuple[1]
        uimage = imtuple[2]
        
        gradi = zeros

        if pol_solve[1]!=0:
            gradq = -np.real(np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradq = zeros

        if pol_solve[2]!=0:
            gradu = -np.imag(np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradu = zeros

        gradout = np.array((gradi, gradq, gradu))


    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_prim)

    return gradout
    
    
    
