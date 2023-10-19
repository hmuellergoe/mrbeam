import numpy as np
from scipy.special import j1

from lightwise import imgutils, nputils

class Sinc2D(imgutils.AbstractBeam):
    def __init__(self, width, nsigma=50, support=None):
        self.width = width
        self.nsigma = nsigma

        if support == None:
            support = self._sinc_support()
        self.support = support
        self.size = nputils.get_pair(self.support)
        x, y = np.indices(self.size, dtype=int)
        center = np.floor(self.size / 2.)

        xp = (x-center[0])**2
        yp = (y-center[1])**2

        self.diff = np.sqrt(xp+yp)/self.width

        super().__init__()

    def __str__(self):
        return "Sinc:Scale:%s" % (self.width)

    def build_beam(self):
        beam = np.sinc(self.diff)
        return beam/np.sum(beam)
        
    def _sinc_support(self):
        return int(np.ceil(2 * self.nsigma * self.width))  

class Bessel2D(imgutils.AbstractBeam):
    def __init__(self, a, b, degree=0, nsigma=50, support=None):
        self.a = a
        self.b = b
        self.nsigma = nsigma

        if support == None:
            support = self._bessel_support()
        self.support = support
        self.size = nputils.get_pair(self.support)
        x, y = np.indices(self.size, dtype=int)
        center = np.floor(self.size / 2.)

        xp = (x-center[0])
        yp = (y-center[1])

        coords = np.zeros((2, xp.shape[0], xp.shape[1]))
        coords[0] = xp
        coords[1] = yp

        #the degree in lightwise is defined clock-wise,
        #we therefore need the inverse rotation matrix here to stay consistent with Gaussian beams
        rotation_matrix = np.array([[np.cos(degree), -np.sin(degree)], [np.sin(degree), np.cos(degree)]])

        coords_rot = np.tensordot(rotation_matrix, coords, axes=1)

        self.diff = np.sqrt( coords_rot[0]**2/self.a**2 + coords_rot[1]**2/self.b**2 )

        super().__init__()

    def __str__(self):
        return "Bessel:Scale:%s%s" % (self.a, self.b)

    def build_beam(self):
        beam = j1(2*np.pi*self.diff)/(self.diff*self.a*self.b)
        beam = np.where(self.diff == 0, np.pi/(self.a*self.b), beam)
        return beam/np.sum(beam)
        
    def _bessel_support(self):
        support = int(np.ceil(2 * self.nsigma * np.sqrt(self.a*self.b)))   
        return int(2 * (support // 2) + 1)

