import numpy as np 
from regpy.discrs import Discretization

#norm=0: normalize to maximal value 1
#norm=1: normalize to flux 1
class Gaussian2D():
    def __init__(self, grid, norm=0):
        assert isinstance(grid, Discretization)
        assert grid.ndim == 2
        self.grid = grid
        assert norm in np.array([0,1])
        self.norm = norm
        
    def gaussianblob(self, mu, semiaxes, degree):
        assert len(mu) == 2
        assert len(semiaxes) == 2
        assert (isinstance(degree, float) or isinstance(degree, int))
        
        a = semiaxes[0]
        b = semiaxes[1]
        
        assert (isinstance(a, float) or isinstance(a, int))
        assert (isinstance(b, float) or isinstance(b, int))
        
        degree = np.pi*degree/180
        cos = np.cos(degree)
        sin = np.sin(degree)
        
        vec = np.zeros((self.grid.shape[0], self.grid.shape[1], 2))
        vec[:, :, 0] = self.grid.coords[0, :, :]
        vec[:, :, 1] = self.grid.coords[1, :, :]
        
        Sigma = np.array([[a**2*cos**2+b**2*sin**2, cos*sin*(a**2-b**2)], [cos*sin*(a**2-b**2), a**2*sin**2+b**2*cos**2]])

        #toret = np.zeros(self.grid.shape)
        #for i in range(self.grid.shape[0]):
        #    for j in range(self.grid.shape[1]):
        #        toret[i, j] = self._gaussian2d(vec[i, j, :], mu, Sigma)
        toret = self._gaussian2d(vec, mu, Sigma)
        if self.norm == 0:
            return toret/np.max(toret)
        else:
            return toret
        
    def _gaussian2d(self, vec, mu, Sigma):
        #exponent = -0.5*np.dot( vec-mu,  np.linalg.inv(Sigma) @ (vec-mu) )
        exponent = -0.5*np.sum( ((vec-mu) @ np.linalg.inv(Sigma))*(vec-mu), axis=2)
        return 1/(2*np.pi*np.sqrt(np.linalg.det(Sigma)))*np.exp(exponent)
        