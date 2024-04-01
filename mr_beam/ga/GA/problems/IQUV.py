import pygmo as pg
import ehtim as eh
import numpy as np

from imagingbase.ehtim_wrapper import EhtimFunctional
from imagingbase.ehtim_wrapper import EhtimWrapper, EmptyFunctional
from imagingbase.ehtim_wrapper_pol import EhtimWrapperPol
from regpy.discrs import Discretization
from regpy.operators import Reshape, CoordinateMask, CoordinateProjection

from joblib import Parallel, delayed

from GA.problems.EHT import MyFunc as EHTFunc
from GA.problems.Full_Stokes import MyFunc as FullStokesFunc

class MyFunc():
    def __init__(self, obs, prior, data_term, reg_term, ttype='direct', rescaling=0.1, rescalingV=0.002, zbl=1, dim=1, mode='pareto'):
        ###Stokes I###
        ehtfunc = EHTFunc(obs, prior, data_term, reg_term, ttype=ttype, rescaling=rescaling, zbl=zbl, dim=dim, mode=mode)
        
        ###Polarization###
        polfunc = FullStokesFunc(obs, prior, data_term, reg_term, ttype=ttype, rescaling=rescaling, rescaling=rescalingV, zbl=zbl, dim=dim, mode=mode)
        
        ###POLARIZATION###
        size = ehtfunc.wrapper.xtuple.shape[1]
        grid = Discretization(4*size)
        
        zeros = np.zeros(grid.shape, dtype=bool)
        
        maskI = zeros.copy()
        maskI[0:size] = True
        self.projI = CoordinateProjection(grid, maskI)
        
        maskQUV = zeros.copy()
        maskQUV[size:] = True
        self.projQUV = CoordinateProjection(grid, maskQUV)
        
        self.ob1 = ehtfunc.ob1 * self.projI
        self.ob2 = ehtfunc.ob2 * self.projI
        self.ob3 = ehtfunc.ob3 * self.projI
        self.ob4 = ehtfunc.ob4 * self.projI
        self.ob5 = ehtfunc.ob5 * self.projI        
        self.ob6 = ehtfunc.ob6 * self.projI
        self.ob7 = ehtfunc.ob7 * self.projI
            
        self.ob8 = polfunc.ob8 * self.projQUV
        self.ob9 = polfunc.ob9 * self.projQUV
        self.ob10 = polfunc.ob10 * self.projQUV
        self.ob11 = polfunc.ob11 * self.projQUV
        self.ob12 = polfunc.ob12 * self.projQUV        

        

class IQUV:
    """
    A multi-objective problem.
    (This is actually a Python implementation of 2-dimensional ZDT-1 problem)

    USAGE: my_mo_problem()
    """
        
    def __init__(self, obs, prior, data_term, reg_term, rescaling, rescalingV, zbl, dim, num_cores=16, ttype='direct', mode='pareto'):
        self.obs = obs
        self.prior = prior
        self.data_term = data_term
        self.reg_term = reg_term
        self.rescaling = rescaling
        self.rescalingV = rescalingV
        self.zbl = zbl
        self.dim = dim
        self.mode = mode
        
        self.num_cores = num_cores
        
    def setFit(self):
        self.fit = MyFunc(self.obs, self.prior, self.data_term, self.reg_term, rescaling=self.rescaling, rescalingV=self.rescalingV, zbl=self.zbl, mode=self.mode)
        return self.fit
    
    def batch_fitness(self, dvs):
        if 'fit' not in self.__dict__:
            self.fit = self.setFit()
        samples = len(dvs) // self.dim
        x = dvs.reshape((samples, self.dim))
              
        results = Parallel(n_jobs=self.num_cores)(delayed(self.fitness)(x[i]) for i in range(samples))
        return np.asarray(results).flatten()
        
    # Reimplement the virtual method that defines the objective function
    def fitness(self, x):
        if 'fit' not in self.__dict__:
            self.fit = self.setFit()
        return [self.fit.ob1(x),self.fit.ob2(x),self.fit.ob3(x),self.fit.ob4(x),self.fit.ob5(x), self.fit.ob6(x), self.fit.ob7(x), self.fit.ob8(x), self.fit.ob9(x), self.fit.ob10(x), self.fit.ob11(x), self.fit.ob12(x)]
    
    def get_nobj(self):
        return 12

    def get_bounds(self):
        return (np.full((self.dim,),0.),np.full((self.dim,),1))


    def gradient(self, x):
        return np.concatenate(( self.fit.ob1.gradient(x), self.fit.ob2.gradient(x), self.fit.ob3.gradient(x), 
                              self.fit.ob4.gradient(x), self.fit.ob5.gradient(x), self.fit.ob6.gradient(x), self.fit.ob7.gradient(x), self.fit.ob8.gradient(x), self.fit.ob9.gradient(x), self.fit.ob10.gradient(x), self.fit.ob11.gradient(x), self.fit.ob12.gradient(x) )) 
        #return estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        return "IQUV"

