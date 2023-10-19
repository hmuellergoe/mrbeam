import numpy as np

from regpy.operators import CoordinateProjection
from regpy.operators import DirectSum, Reshape, Zero
from regpy.discrs import Discretization
from regpy.solvers import HilbertSpaceSetting
from regpy.hilbert import L2

from imagingbase.ehtim_wrapper import EhtimFunctional
from imagingbase.ehtim_wrapper_pol import EhtimWrapperPol

from MSI.Image import ConversionBase



class PolImager():
    def __init__(self, handler):
        self.handler = handler
        return
    
    def flag_obs(self, obs_sc):
        if np.isnan(np.sum(obs_sc.data['qvis'])):
            flagged_obs = obs_sc.copy()
            indices = np.isnan(obs_sc.data['qvis'])
            for j in range(len(obs_sc.data)):
                if indices[j]:
                    bl = [obs_sc.data['t1'][j], obs_sc.data['t2'][j]]
                    flagged_obs = flagged_obs.flag_bl(bl)
            obs_sc = flagged_obs.copy()
            
        return obs_sc
    
    def init_polarimetry(self, coeff, obs_sc, pol_frac=-0.01, cinit=True, initimg=None):
        init = self.handler.coordinate_proj(coeff)
        if initimg==None:
            initimg = self.handler.wrapper.formatoutput(self.handler.op(coeff))
        coeff_pol = self.handler.coordinate_proj.adjoint(init)
        
        wrapper = EhtimWrapperPol(obs_sc, initimg, initimg, self.handler.zbl, d='pvis', clipfloor=-100, pol_solve=(1,1,1), pol_trans=False)
        pol = PolHandler(wrapper, self.handler.mask, self.handler.op, coeff_pol*self.handler.rescaling)
    
        if cinit:
            init = np.zeros((3, np.sum(self.handler.mask)))
            init[1] = pol_frac * pol.coordinate_proj(coeff_pol*self.handler.rescaling)
            init = init.flatten()
        
            return [init, pol]
        
        else: 
            return pol
    
    def init_circular_pol(self, coeff, obs, initimg=None, cinit=True):
        init = self.handler.coordinate_proj(coeff)
        if initimg==None:
            initimg = self.handler.wrapper.formatoutput(self.handler.op(coeff))
        coeff_pol = self.handler.coordinate_proj.adjoint(init)
        wrapperV = EhtimWrapperPol(obs, initimg, initimg, self.handler.zbl, d='pvis', clipfloor=-100, pol_solve=(1,1,1), pol_trans=False, stokesv=True)
        polV = PolHandler(wrapperV, self.handler.mask, self.handler.op, coeff_pol*self.handler.rescaling)
        initv = np.zeros(polV.vfunc.domain.shape)
         
        if cinit:
            return [initv, polV]
        else:
            return polV

class PolHandler():
    def __init__(self, wrapper, mask, dictionary, init):
        self.wrapper = wrapper
        self.func = EhtimFunctional(self.wrapper, Discretization(self.wrapper.xtuple.shape))
                    
        self.inittuple = self.wrapper.inittuple
        self.mask = np.concatenate((mask, mask))
        self.op = dictionary
        
        self.npix = self.wrapper.InitIm.xdim
        
        self.convert = ConversionBase()
        
        #Define domain
        self.convert = ConversionBase()
        self.grid = self.convert.find_domain_ehtim(self.wrapper.InitIm)
        
        self.length = len(mask) // (self.npix**2)
        
        self.domain = self.grid**(self.length)
        
        self.coordinate_proj = CoordinateProjection(self.domain, mask)
        
        #projection for Q and U
        self.domain_proj = self.domain**2
        self.proj = CoordinateProjection(self.domain_proj, self.mask)
        
        #Q and U wavelet dictionary
        self.msi = DirectSum(self.op, self.op)
        self.op_proj = self.msi * Reshape(self.proj.domain, self.msi.domain) * self.proj.adjoint

        #Add Stokes I to array
        self.zero_op = Zero(Discretization(np.sum(mask)), Discretization(self.npix**2))
        self.op_proj_final = DirectSum(self.zero_op, self.op_proj)
        
        self.shift = np.zeros((3, self.npix**2))
        self.shift[0] = self.inittuple[0]
        self.shift = self.shift.flatten()

        # Construct final functionals
        self.final_op_shifted = self.op_proj_final + self.shift
        self.final_op = Reshape(self.final_op_shifted.codomain, self.func.domain) * self.final_op_shifted
        self.final_func = self.func * self.final_op
        
        self.final_init = np.zeros((3, np.sum(mask)))
        self.final_init[1] = 0.2 * self.coordinate_proj(init)
        self.final_init = self.final_init.flatten()

        # empty setting class
        self.setting_op = self.final_func.domain.identity
        self.final_setting = HilbertSpaceSetting(self.setting_op, L2, L2)
        
        # functionals for Stokes V imaging
        self.vproj = CoordinateProjection(self.op.domain, mask)
        self.vdomain = Discretization(self.op.codomain.size)
        self.vfunc = EhtimFunctional(self.wrapper, self.vdomain) * Reshape(self.op.codomain, self.vdomain) * self.op *  self.vproj.adjoint
        self.vsetting = HilbertSpaceSetting(self.vproj.adjoint, L2, L2)

    def find_pol_init(self, img):
        iimage = img.get_polvec('i')
        qimage = img.get_polvec('q')
        uimage = img.get_polvec('u')

        qinit = np.zeros((self.length, self.npix**2))
        uinit = np.zeros((self.length, self.npix**2))
        
        for i in range(self.length):
            qinit[i] = qimage
            uinit[i] = uimage
            
        self.final_init = np.zeros((3, self.final_init.shape[0]//3))
        self.final_init[1] = self.coordinate_proj(qinit.flatten())
        self.final_init[2] = self.coordinate_proj(uinit.flatten())
        self.final_init = self.final_init.flatten()

        self.shift = np.zeros((3, self.npix**2))
        self.shift[0] = iimage
        self.shift = self.shift.flatten()

        # Construct final functionals
        self.final_op_shifted = self.op_proj_final + self.shift
        self.final_op = Reshape(self.final_op_shifted.codomain, self.func.domain) * self.final_op_shifted
        self.final_func = self.func * self.final_op








































