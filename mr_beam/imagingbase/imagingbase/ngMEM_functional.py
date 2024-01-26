import numpy as np
from regpy.functionals import Functional
from imagingbase.ehtim_wrapper import EhtimWrapper, EhtimFunctional

class MovieEntropy(Functional):
    def __init__(self, handler, domain, prior_movie):
        self.handler = handler
        self.nr_of_frames = len(prior_movie)
        
        self.handlers = []
        self.functionals = []
        for i in range(self.nr_of_frames):
            img = self.handler.formatoutput(prior_movie[i])
            self.handlers.append(EhtimWrapper(self.handler.Obsdata, img, img, img.total_flux(),
                                              d='simple', ttype=self.handler.ttype, clipfloor=self.handler.clipfloor))
            self.functionals.append(EhtimFunctional(self.handlers[i], domain))
        super().__init__(domain**self.nr_of_frames)
    
    def _eval(self, imvecs):
        _imvecs = self.domain.split(imvecs)
        toret = 0
        for i in range(self.nr_of_frames):
            toret += self.functionals[i](_imvecs[i])
        return toret
    
    def _gradient(self, imvecs):
        _imvecs = self.domain.split(imvecs)
        toret = 0
        for i in range(self.nr_of_frames):
            toret += self.functionals[i].gradient(_imvecs[i])
        return toret
    
class TemporalEntropy(Functional):
    def __init__(self, domain, nr_of_frames, C, tau):
        self.nr_of_frames = nr_of_frames
        self.C = C
        self.tau = tau
        self.times = np.arange(self.nr_of_frames)
        
        self.tdiff = np.zeros((self.nr_of_frames, self.nr_of_frames))
        for i in range(self.nr_of_frames):
            for j in range(self.nr_of_frames):
                self.tdiff[i,j] = np.abs(self.times[i]-self.times[j])
        self.imvecdiff = np.zeros((domain.size, self.nr_of_frames, self.nr_of_frames))
        super().__init__(domain**self.nr_of_frames)
    
    def _eval(self, imvecs):
        _imvecs = self.domain.split(imvecs)
        for i in range(self.nr_of_frames):
            for j in range(self.nr_of_frames):
                self.imvecdiff[:,i,j] = np.abs(_imvecs[i].flatten()-_imvecs[j].flatten())
        
        return np.sum(np.exp(-self.tdiff**2/(2*self.tau**2))*self.imvecdiff*np.log(self.imvecdiff+self.C))
        
    def _gradient(self, imvecs):
        _imvecs = self.domain.split(imvecs)
        for i in range(self.nr_of_frames):
            for j in range(self.nr_of_frames):
                self.imvecdiff[:,i,j] = np.abs(_imvecs[i].flatten()-_imvecs[j].flatten())
                
        return np.sum(np.exp(-self.tdiff**2/(2*self.tau**2))*(np.log(self.imvecdiff+self.C)+self.imvecdiff/(self.imvecdiff+self.C)), axis=1).flatten() \
                    + np.sum(np.exp(-self.tdiff**2/(2*self.tau**2))*(np.log(self.imvecdiff+self.C)+self.imvecdiff/(self.imvecdiff+self.C)), axis=2).flatten()
    
    
    
    
    
    
    
        