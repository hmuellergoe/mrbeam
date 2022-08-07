from astropy.io import fits
from lightwise import imgutils
import ehtim.const_def as ehc
from regpy.discrs import UniformGrid
import numpy as np

class FitsBase():
    def __init__(self): 
        self.hdul = None
        return
    
    def load_fits_from_path(self, path):
        self.hdul = fits.open(path)
        return
        
    def get_fits_data(self):  
        if self.hdul ==None:
            print("Load fits file to base first")
            return
        return self.hdul[0].data[0, 0]
    
    def close_fits(self):
        if self.hdul ==None:
            print("Load fits file to base first")
            return
        self.hdul.close()
        return
    
class ConversionBase():
    def __init__(self):
        return
    
    def ehtim_to_numpy(self, im):
        toret = np.reshape(im.imvec, (im.ydim, im.xdim))[:, :]
        return toret
    
    def numpy_to_ehtim(self, data, im):
        data = data[:, :].flatten()
        im.imvec = data
        return im
        
    def numpy_to_libwise(self, data):
        return imgutils.Image(data)
    
    def libwise_to_numpy(self, im):
        return im.data
    
    def ehtim_to_libwise(self, im):
        data = self.ehtim_to_numpy(im)
        return self.numpy_to_libwise(data)
    
    def libwise_to_ehtim(self, im_lib, im_eht):
        data = self.libwise_to_numpy(im_lib)
        return self.numpy_to_ehtim(data, im_eht)
    
    def find_domain_ehtim(self, im, dtype=float):
        psize = im.psize / ehc.RADPERUAS
        grid = UniformGrid((-psize/2*im.xdim, psize/2*im.xdim, im.xdim), (-psize/2*im.ydim, psize/2*im.ydim, im.ydim), dtype=dtype)
        return grid
        #return [psize, im.xdim, im.ydim]
        