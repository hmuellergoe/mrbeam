from lightwise import imgutils, wtutils, nputils
from MSI.MSDecomposition import Beam

from MSI.utils.beams import Bessel2D as bessel

import numpy as np

class DOGDictionary():
    def __init__(self, widths, shape, angle=0, ellipticities=6, nsigma=10, support=None, rml=False, **args):
        self.widths = widths
        self.angle = angle
        self.angles = np.linspace(0, np.pi, ellipticities+1)[:-1] + self.angle
        self.ellipticities = ellipticities
        self.shape = shape
        
        self.nsigma = nsigma
        self.support = support
        self.rml = rml
        
        if self.rml:
            self.length = (len(self.widths)-1)*(self.ellipticities+1)+1
        else:
            self.length = (len(self.widths))*(self.ellipticities+1)
        
        self.dirac = np.zeros(self.shape)
        self.dirac[self.shape[0]//2, self.shape[1]//2] = 1      
        
    #Computes scaling functions as imgutils.GaussianBeam object and not as numpy,
    #thus only useful for inner computation in the MSI module
    def _compute_scaling_functions(self):
        
        beams_ell = []
        for i in range(len(self.widths)-1):
            w_small = self.widths[i]
            w_large = self.widths[i+1]
            beams = []
            for degree in self.angles:
                beams.append(imgutils.GaussianBeam(w_large, w_small, bpa=degree, nsigma=self.nsigma))  
            beams_ell.append(beams)
        beams_circ = [imgutils.GaussianBeam(w, w, bpa=self.angle, nsigma=self.nsigma) for w in self.widths]

        return [beams_ell, beams_circ]
    
    def compute_dogs(self):
        beams_ell, beams_circ = self._compute_scaling_functions()
        
        filtered_ell = []
        for i in range(len(beams_ell)):
            filtered_ell_one_width = []
            for j in range(len(self.angles)):
                filtered_ell_one_width.append(beams_ell[i][j].convolve(self.dirac))
            filtered_ell.append(filtered_ell_one_width)

        filtered_circ = [b.convolve(self.dirac) for b in beams_circ]
        
        list_wavelets = []
        if self.rml:
            for i in range(len(beams_ell)):
                remainder = len(self.angles) * filtered_circ[i]
                for j in range(len(self.angles)):
                    list_wavelets.append(filtered_ell[i][j]-filtered_circ[i+1])
                    remainder -= filtered_ell[i][j]
                list_wavelets.append(remainder)

            list_wavelets.append( len(self.angles) * filtered_circ[-1] )
            
        else:
            for i in range(len(beams_ell)):
                remainder = len(self.angles) * filtered_circ[i]
                for j in range(len(self.angles)):
                    list_wavelets.append(filtered_ell[i][j]-filtered_circ[i+1])
                    remainder -= filtered_ell[i][j]
    #            list_wavelets.append(remainder)
                list_wavelets.append(remainder / len(self.angles))
            
            remainder = len(self.angles) * filtered_circ[-1]
            for j in range(len(self.angles)):
                list_wavelets.append(filtered_ell[-1][j])
                remainder -= filtered_ell[-1][j]
    #        list_wavelets.append(remainder)
    #        list_wavelets.append(remainder / len(self.angles))
            list_wavelets.append( filtered_circ[-1] )
        
        return list_wavelets

    def set_smoothing_scale(self, list_wavelets, width):
        del(list_wavelets[-1])
        beam = imgutils.GaussianBeam(width, width, bpa=self.angle, nsigma=self.nsigma)
        list_wavelets.append(beam.convolve(self.dirac))
        return list_wavelets

class BesselDictionary():
    def __init__(self, widths, shape, angle=0, ellipticities=6, nsigma=50, support=None, rml=False, **args):
        self.widths = widths
        self.angle = angle
        self.angles = np.linspace(0, np.pi, ellipticities+1)[:-1] + self.angle
        self.ellipticities = ellipticities
        self.shape = shape
        
        self.nsigma = nsigma
        self.support = support
        self.rml = rml
        
        if self.rml:
            self.length = (len(self.widths)-1)*(self.ellipticities+1)+1
        else:
            self.length = (len(self.widths))*(self.ellipticities+1)
        
        self.dirac = np.zeros(self.shape)
        self.dirac[self.shape[0]//2, self.shape[1]//2] = 1      
        
    #Computes scaling functions as imgutils.GaussianBeam object and not as numpy,
    #thus only useful for inner computation in the MSI module
    def _compute_scaling_functions(self):
        
        beams_ell = []
        for i in range(len(self.widths)-1):
            w_small = self.widths[i]
            w_large = self.widths[i+1]
            beams = []
            for degree in self.angles:
                beams.append(bessel(w_large, w_small, degree=degree, nsigma=self.nsigma, support=self.support))  
            beams_ell.append(beams)
        beams_circ = [bessel(w, w, degree=0, nsigma=self.nsigma, support=self.support) for w in self.widths]

        return [beams_ell, beams_circ]
    
    def compute_wavelets(self):
        beams_ell, beams_circ = self._compute_scaling_functions()
        
        filtered_ell = []
        for i in range(len(beams_ell)):
            filtered_ell_one_width = []
            for j in range(len(self.angles)):
                filtered_ell_one_width.append(beams_ell[i][j].convolve(self.dirac))
            filtered_ell.append(filtered_ell_one_width)

        filtered_circ = [b.convolve(self.dirac) for b in beams_circ]
        
        list_wavelets = []
        if self.rml:
            for i in range(len(beams_ell)):
                remainder = len(self.angles) * filtered_circ[i]
                for j in range(len(self.angles)):
                    list_wavelets.append(filtered_ell[i][j]-filtered_circ[i+1])
                    remainder -= filtered_ell[i][j]
                list_wavelets.append(remainder)

            list_wavelets.append( len(self.angles) * filtered_circ[-1] )
        else:
            for i in range(len(beams_ell)):
                remainder = len(self.angles) * filtered_circ[i]
                for j in range(len(self.angles)):
                    list_wavelets.append(filtered_ell[i][j]-filtered_circ[i+1])
                    remainder -= filtered_ell[i][j]
    #            list_wavelets.append(remainder)
                list_wavelets.append(remainder / len(self.angles))
    #            list_wavelets.append( remainder / np.linalg.norm(remainder) * np.linalg.norm(list_wavelets[-1]) )
            
            remainder = len(self.angles) * filtered_circ[-1]
            for j in range(len(self.angles)):
                list_wavelets.append(filtered_ell[-1][j])
                remainder -= filtered_ell[-1][j]
    #        list_wavelets.append(remainder)
    #        list_wavelets.append(remainder / len(self.angles))
            list_wavelets.append( filtered_circ[-1] )
        
        return list_wavelets

    def set_smoothing_scale(self, list_wavelets, width):
        del(list_wavelets[-1])
        beam = bessel(width, width, degree=0, nsigma=self.nsigma, support=self.support)
        list_wavelets.append(beam.convolve(self.dirac))
        return list_wavelets