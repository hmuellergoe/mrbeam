import numpy as np

from regpy.operators import CoordinateProjection, DirectSum, Zero
from regpy.discrs import Discretization
from regpy.solvers import HilbertSpaceSetting
from regpy.hilbert import L2

from imagingbase.ehtim_wrapper import EhtimFunctional
from imagingbase.regpy_utils import Reshape, power

from MSI.Image import ConversionBase

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
        
        self.domain = power(self.grid, self.length)
        
        self.coordinate_proj = CoordinateProjection(self.domain, mask)
        
        #projection for Q and U
        self.domain_proj = power(self.domain, 2)
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









































