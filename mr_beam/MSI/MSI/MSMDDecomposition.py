from lightwise import imgutils, wtutils, nputils
from MSI.MSDecomposition import Beam

from MSI.utils.beams import Bessel2D as bessel

import numpy as np

class DoG2D():
    def __init__(self, widths, angle=0, ellipticities=6, all_scales=False, smoothing_scale=None, nsigma=None, support=None, **args):
        self.angle = angle
        self.angles = np.linspace(0, np.pi, ellipticities+1)[:-1] + self.angle
        self.widths = widths
        self.all_scales = all_scales
        
        self.nsigma=nsigma
        if nsigma==None:
            self.nsigma = 4
        
        if self.all_scales:
            self.smoothing_scale = smoothing_scale or self.widths[-1]
            self.update_smoothing_scale = False
            if self.smoothing_scale != self.widths[-1]:
                self.update_smoothing_scale = True
                self.smoothing_beam = imgutils.GaussianBeam(self.smoothing_scale, self.smoothing_scale, bpa=self.angle, nsigma=self.nsigma)
        self.bg = 1
        self.beams_ell = []
        for i in range(len(self.widths)-1):
            w_small = self.widths[i]
            w_large = self.widths[i+1]
            beams = []
            for degree in self.angles:
                beams.append(imgutils.GaussianBeam(w_large, w_small, bpa=degree, nsigma=self.nsigma))  
            self.beams_ell.append(beams)
        self.beams_circ = [imgutils.GaussianBeam(w, w, bpa=self.angle, nsigma=self.nsigma) for w in self.widths]
                
    def decompose(self, img):
        img = img.copy()
        filtered_ell = []
        for i in range(len(self.beams_ell)):
            filtered_ell_one_width = []
            for j in range(len(self.angles)):
                filtered_ell_one_width.append(self.beams_ell[i][j].convolve(img))
            filtered_ell.append(filtered_ell_one_width)

        filtered_circ = [b.convolve(img) for b in self.beams_circ]

        res = []
        for i in range(len(self.beams_ell)):
            remainder = len(self.angles) * filtered_circ[i]
            for j in range(len(self.angles)):
                res.append(filtered_ell[i][j]-filtered_circ[i+1])
                remainder -= filtered_ell[i][j]
#            res.append(remainder)
            res.append(remainder / len(self.angles))

        if self.all_scales:
            if self.update_smoothing_scale:
                res.append( len(self.angles) * self.smoothing_beam.convolve(img) )
            else:
                remainder = len(self.angles) * filtered_circ[-1]
                for j in range(len(self.angles)):
                    res.append(filtered_ell[-1][j])
                    remainder -= filtered_ell[-1][j]
                res.append( filtered_circ[-1] )
#                res.append(remainder)
#                res.append(remainder / len(self.angles))
        
        toret = []
        
        decomposed = zip(res, np.arange(0, len(res)))
        
        for scale, width in decomposed:
            toret.append([scale, width])
            
        return toret

    def addbeam(self, data):
        for i in range(len(self.widths)-1):
            for j in range(len(self.angles)):
                b = self.beams_ell[i][j]
                self.beams_ell[i][j] = Beam(b.convolve(data))
        self.beams_circ = [Beam(b.convolve(data)) for b in self.beams_circ]
        if self.all_scales and self.update_smoothing_scale:
            self.smoothing_beam = Beam(self.smoothing_beam.convolve(data))


class Bessel2D():
    def __init__(self, widths, angle=0, ellipticities=6, all_scales=False, nsigma=10, support=None, smoothing_scale=None, **args):
        self.nsigma = nsigma
        self.support = support
        
        self.angle = angle
        self.angles = np.linspace(0, np.pi, ellipticities+1)[:-1] + self.angle
        self.widths = widths
        self.all_scales = all_scales
        if self.all_scales:
            self.smoothing_scale = smoothing_scale or self.widths[-1]
            self.update_smoothing_scale = False
            if self.smoothing_scale != self.widths[-1]:
                self.update_smoothing_scale = True
                self.smoothing_beam = bessel(self.smoothing_scale, self.smoothing_scale, degree=self.angle, nsigma=self.nsigma, support=self.support)
        self.bg = 1
        self.beams_ell = []
        for i in range(len(self.widths)-1):
            w_small = self.widths[i]
            w_large = self.widths[i+1]
            beams = []
            for degree in self.angles:
                beams.append(bessel(w_large, w_small, degree=degree, nsigma=self.nsigma, support=self.support))  
            self.beams_ell.append(beams)
        self.beams_circ = [bessel(w, w, degree=self.angle, nsigma=self.nsigma, support=self.support) for w in self.widths]
                
    def decompose(self, img):
        img = img.copy()
        filtered_ell = []
        for i in range(len(self.beams_ell)):
            filtered_ell_one_width = []
            for j in range(len(self.angles)):
                filtered_ell_one_width.append(self.beams_ell[i][j].convolve(img))
            filtered_ell.append(filtered_ell_one_width)

        filtered_circ = [b.convolve(img) for b in self.beams_circ]

        res = []
        for i in range(len(self.beams_ell)):
            remainder = len(self.angles) * filtered_circ[i]
            for j in range(len(self.angles)):
                res.append(filtered_ell[i][j]-filtered_circ[i+1])
                remainder -= filtered_ell[i][j]
#            res.append(remainder)
            res.append(remainder / len(self.angles))

        if self.all_scales:
            if self.update_smoothing_scale:
                res.append( len(self.angles) * self.smoothing_beam.convolve(img) )
            else:
                remainder = len(self.angles) * filtered_circ[-1]
                for j in range(len(self.angles)):
                    res.append(filtered_ell[-1][j])
                    remainder -= filtered_ell[-1][j]
                res.append( filtered_circ[-1] )
#                res.append(remainder)
#                res.append(remainder / len(self.angles))
        
        toret = []
        
        decomposed = zip(res, np.arange(0, len(res)))
        
        for scale, width in decomposed:
            toret.append([scale, width])
            
        return toret

    def addbeam(self, data):
        for i in range(len(self.widths)-1):
            for j in range(len(self.angles)):
                b = self.beams_ell[i][j]
                self.beams_ell[i][j] = Beam(b.convolve(data))
        self.beams_circ = [Beam(b.convolve(data)) for b in self.beams_circ]
        if self.all_scales and self.update_smoothing_scale:
            self.smoothing_beam = Beam(self.smoothing_beam.convolve(data))