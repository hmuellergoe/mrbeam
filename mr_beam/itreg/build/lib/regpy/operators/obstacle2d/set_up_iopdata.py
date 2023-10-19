import numpy as np
import scipy.linalg as scla

from scipy.special import j0, j1, y0, y1

def setup_iop_data(bd,kappa):
    """ computes data needed to set up the boundary integral matrices
     to avoid repeated computations"""
    dimension = len(bd.z.shape)
    dim=np.max([np.size(bd.z, l) for l in range(0, dimension)])
#    dat.kappa = kappa

    """compute matrix of distances of grid points"""
    t1=np.matlib.repmat(bd.z[0,:].T,1,dim).reshape(dim**2)-np.matlib.repmat(bd.z[0,:], dim, 1).reshape(dim**2)
    t2=np.matlib.repmat(bd.z[1,:].T,1,dim).reshape(dim**2)-np.matlib.repmat(bd.z[1,:], dim, 1).reshape(dim**2)
    kdist = kappa*np.sqrt(t1**2 + t2**2)
    bessj0_kdist = j0(kdist)
    bessH0 = bessj0_kdist + complex(0, 1) * y0(kdist, bessj0_kdist)
    bessH0=bessH0.reshape((dim, dim))
    #bessH0 = besselh(0,1,dat.kdist)

    bessj1_kdist= j1(kdist)
    """bessH1quot=np.zeros(dim**2)
    for i in range(0, dim):
        for j in range(0, dim):
            if kdist[dim*i+j]==0:
                bessH1quot[dim*i+j]=0
            else:
                bessH1quot[dim*i+j] = (bessj1_kdist[dim*i+j]+ complex(0,1)*bessy1(kdist[dim*i+j],bessj1_kdist[dim*i+j]))/kdist[dim*i+j]"""
    bessH1quot = (bessj1_kdist + complex(0,1) * y1(kdist, bessj1_kdist)) / (kdist + 1e-5)
    bessH1quot=bessH1quot.reshape((dim, dim))
    #bessH1quot = besselh(1,1,dat.kdist) / dat.kdist
    for j in range(0, dim):
        bessH0[j,j]=1


    """set up prototyp of the singularity of boundary integral operators"""
    t=2*np.pi*np.arange(1, dim)/dim
    logsin = scla.toeplitz(np.append(np.asarray([1]), np.log(4*np.sin(t/2)**2)))

    """quadrature weight for weight function log(4*(sin(t-tau))**2)"""
    sign=np.ones(dim)
    sign[np.arange(1, dim, 2)]=-1
    t = 2*np.pi*np.arange(0, dim)/dim
    s=0
    for m in range(0, int(dim/2)-1):
        s=s+np.cos((m+1)*t)/(m+1)
    logsin_weights = scla.toeplitz(-2*(s + sign/dim)/dim)

    #euler constant 'eulergamma'
    Euler_gamma =  0.577215664901532860606512

    kdist=kdist.reshape((dim, dim))

    return dat_object(kappa, Euler_gamma, logsin_weights, logsin, bessH0, bessH1quot, \
                      kdist )


class dat_object(object):
    def __init__(self, kappa, Euler_gamma, logsin_weights, logsin, bessH0, bessH1quot, \
                      kdist):
        self.kappa=kappa
        self.Euler_gamma=Euler_gamma
        self.logsin_weights=logsin_weights
        self.logsin=logsin
        self.bessH0=bessH0
        self.bessH1quot=bessH1quot
        self.kdist=kdist
