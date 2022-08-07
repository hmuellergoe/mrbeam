from lightwise import imgutils, wtutils, nputils
from MSI.utils.beams import Bessel2D as bessel

import numpy as np

'''
Possible Parameters:
wavelet_fct: 'b1', 'b3', 'b5', 'db1'-'db18', 'sym2'-'sym8', 'coif1'-'coif5', 'triangle', 'triangle2' 
wt_dec: uiwt, uimwt, dwt, uwt
'''    

class WaveletTransform2D():    
    def __init__(self, wavelet_fct='b1', wt_dec = 'uiwt', min_scale=1, max_scale=4, all_scales=False, bg=None, beam=None, **args):
        self.wavelet_fct = wavelet_fct
        if wt_dec == 'uiwt':
            self.wt_dec = wtutils.uiwt
        elif wt_dec == 'uiwmt':
            self.wt_dec = wtutils.uiwmt            
        elif wt_dec == 'dwt':
            self.wt_dec = wtutils.dwt            
        elif wt_dec == 'uwt':
            self.wt_dec = wtutils.uwt
        else:
            raise NotImplementedError()
        self.min_scale = min_scale
        self.max_scale = max_scale  
        self.all_scales = all_scales
        self.bg = bg or 1
        self.beam = beam
        
        if self.wavelet_fct in ['b3', 'triangle2']:
            self.scales_width = [max(1.5, 3 * min(1, j) * pow(2, max(0, j - 1))) for j in range(self.min_scale, self.max_scale)]
        else:
            self.scales_width = [max(1, 2 * min(1, j) * pow(2, max(0, j - 1))) for j in range(self.min_scale, self.max_scale)]

        if self.all_scales:
            self.scales_width.append(2**self.max_scale)

    def decompose(self, img, compute_noise=False):
        scales = wtutils.wavedec(img, self.wavelet_fct, self.max_scale, dec=self.wt_dec)
        #self.approx = scales[-1]
        if self.all_scales == False:
            scales = scales[self.min_scale:-1]
            
        scales = [nputils.resize_like(s, img) for s in scales]
        
        toret = []
        
        if compute_noise:
            scales_noise = wtutils.wave_noise_factor(self.bg, self.wavelet_fct, self.max_scale, self.wt_dec, beam=self.beam)
            scales_noise = scales_noise[self.min_scale:]      
    
            decomposed = zip(scales, self.scales_width, scales_noise)
    
            for scale, width, scales_noise in decomposed:
                toret.append([scale, width, scales_noise])
        
        else:
            decomposed = zip(scales, self.scales_width)
            
            for scale, width in decomposed:
                toret.append([scale, width])

        return toret
    
    def compute_noise(self):
        assert self.beam is not None
        
        scales_noise = wtutils.wave_noise_factor(self.bg, self.wavelet_fct, self.max_scale, self.wt_dec, beam=self.beam)
        scales_noise = scales_noise[self.min_scale:] 
        
        decomposed = zip(self.scales_width, scales_noise)
        
        toret = []
        
        for width, scales_noise in decomposed:
            toret.append([width, scales_noise])
            
        return toret

    def compute_scale(self, img, level, axis=None):
        _, d = self.wt_dec(img, self.wavelet_fct, 'symm', level, axis=axis)
        return d

  
class DoG2D():
    def __init__(self, widths, angle=0, ellipticities=1, all_scales=False, smoothing_scale=None, beam=None, nsigma=None, support=None, **args):
        self.angle = angle
        self.ellipticity = ellipticities
        self.widths = widths
        self.all_scales = all_scales
        self.beam = beam
        
        self.nsigma=nsigma
        if nsigma==None:
            self.nsigma = 50
        
        if self.all_scales:
            self.smoothing_scale = smoothing_scale or self.widths[-1]
            self.update_smoothing_scale = False
            if self.smoothing_scale != self.widths[-1]:
                self.update_smoothing_scale = True
                self.smoothing_beam = imgutils.GaussianBeam(self.ellipticity * self.smoothing_scale, self.smoothing_scale, bpa=self.angle, nsigma=self.nsigma)
        self.bg = 1  
        self.beams = [imgutils.GaussianBeam(self.ellipticity * w, w, bpa=self.angle, nsigma=self.nsigma) for w in self.widths]
                
    def decompose(self, img):
        img = img.copy()
        filtered = [b.convolve(img) for b in self.beams]
        res = [(el[0] - el[-1]) for el in nputils.nwise(filtered, 2)]
        
        widths = self.widths

        if self.all_scales:
            if self.update_smoothing_scale:
                res.append(self.smoothing_beam.convolve(img))
            else:
                res.append(filtered[-1])
                
#            delta_cmpnt = [filtered[0]]
#            res = np.concatenate((delta_cmpnt, res))
            
#            widths = np.concatenate(([self.widths[0]], self.widths))
            
        toret = []
        
        decomposed = zip(res, widths)
        
        for scale, width in decomposed:
            toret.append([scale, width])
            
        return toret
    
    def compute_noise(self, shape):
        assert self.beam is not None
        
        #scales_noises = wtutils.dog_noise_factor(self.bg, widths=self.widths, angle=self.angle, ellipticity=self.ellipticity, beam=self.beam)
        bg = nputils.gaussian_noise(shape, 0, self.bg)
        bg = self.beam.convolve(bg)
        
        scales_noises = self.decompose(bg)

        toret = []

        for scales_noise in scales_noises:
            toret.append([np.std(scales_noise[0]), scales_noise[1]])
            
        return toret

    def addbeam(self, data):
        self.beams = [Beam(b.convolve(data)) for b in self.beams]
        if self.all_scales and self.update_smoothing_scale:
            self.smoothing_beam = Beam(self.smoothing_beam.convolve(data))

class Beam():
    def __init__(self, data):
        self.data = data
        
    def convolve(self, img):
        return nputils.convolve(img, self.data, mode='same', boundary="zero")

class Bessel2D():
    def __init__(self, widths, angle=0, ellipticities=1, nsigma=10, all_scales=False, support=None, smoothing_scale=None, **args):
        self.angle = angle
        self.ellipticity = ellipticities        
        self.widths = widths
        self.nsigma = nsigma
        self.beams = [bessel(self.ellipticity * w, w, degree=self.angle, nsigma=self.nsigma, support=support) for w in self.widths]
        self.all_scales = all_scales
        if self.all_scales:
            self.smoothing_scale = smoothing_scale or self.widths[-1]
            self.update_smoothing_scale = False
            if self.smoothing_scale != self.widths[-1]:
                self.update_smoothing_scale = True
                self.smoothing_beam = bessel(self.ellipticity * self.smoothing_scale, self.smoothing_scale, degree=self.angle, nsigma=self.nsigma, support=support)                

    def decompose(self, img):
        img = img.copy()
        filtered = [b.convolve(img) for b in self.beams]
        res = [(el[0]-el[-1]) for el in nputils.nwise(filtered, 2)]
        
        if self.all_scales:
            if self.update_smoothing_scale:
                res.append(self.smoothing_beam.convolve(img))
            else:
                res.append(filtered[-1])

        toret = []

        decomposed = zip(res, self.widths)

        for scale, width in decomposed:
            toret.append([scale, width])

        return toret        

    def addbeam(self, data):
        self.beams = [Beam(b.convolve(data)) for b in self.beams]
        if self.all_scales and self.update_smoothing_scale:
            self.smoothing_beam = Beam(self.smoothing_beam.convolve(data))    
