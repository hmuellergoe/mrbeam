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

class MyFunc(EHTFunc):
    def __init__(self, obs, prior, data_term, reg_term, ttype='direct', rescalingI=0.02, rescaling=0.1, zbl=1, dim=1, mode='pareto'):
        ###Stokes I###
        super().__init__(obs, prior, data_term, reg_term, ttype=ttype, rescaling=rescalingI, zbl=zbl, dim=dim, mode=mode)
        
        ###POLARIZATION###
        #rescaling = 1
        self.wrapper = EhtimWrapperPol(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='pvis', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling*rescalingI, debias=False, pol_solve=(1,1,1), pol_trans=True)
        
        domain = Discretization(self.wrapper.xtuple.shape)
        grid = Discretization(np.prod(domain.shape))
        
        self.func_pvis = EhtimFunctional(self.wrapper, domain)
        self.data_fidelity_term = data_term['pvis'] * self.func_pvis
        self.data_fidelity_term = self.data_fidelity_term * Reshape(grid, domain)
        
        #Add Stokes I to array
        mask_pol = np.zeros(grid.size, dtype='bool')
        mask_pol[self.wrapper.xtuple.shape[1]:] = True
        mask_I = np.ones(grid.size, dtype='bool')
        mask_I[self.wrapper.xtuple.shape[1]:] = False

        proj_pol = CoordinateProjection(self.data_fidelity_term.domain, mask_pol)
        proj_I = CoordinateProjection(self.data_fidelity_term.domain, mask_I)
        mask_pol_op = CoordinateMask(self.data_fidelity_term.domain, mask_pol)
        mask_I_op = CoordinateMask(self.data_fidelity_term.domain, mask_I)
        #no op needed, functional takes correct shape already
        
        #Reshape, rescale fraction and phase of pol
        mask_phase = np.zeros(grid.size, dtype='bool')
        mask_phase[2*self.wrapper.xtuple.shape[1]:] = True
        
        proj_phase = CoordinateProjection(self.data_fidelity_term.domain, mask_phase)
        mask_phase_op = CoordinateMask(self.data_fidelity_term.domain, mask_phase)
        op = (2 * np.pi - 1) * mask_phase_op + mask_pol_op + mask_I_op
        
        self.data_fidelity_term = self.data_fidelity_term * op
        
        prior_floor = prior.copy()
        self.floor = 1/(prior.xdim*prior.ydim)*np.max(prior.imvec) * 1/rescaling
        prior_floor.imvec += 0*self.floor
        
        self.wrapper_msimple = EhtimWrapperPol(obs.copy(), prior_floor.copy(), prior_floor.copy(), zbl,
                            d='msimple', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling*rescalingI, debias=False, pol_solve=(1,1,1), pol_trans=True)

        self.wrapper_hw = EhtimWrapperPol(obs.copy(), prior_floor.copy(), prior_floor.copy(), zbl,
                            d='hw', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling*rescalingI, debias=False, pol_solve=(1,1,1), pol_trans=True)
        
        self.wrapper_ptv = EhtimWrapperPol(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='ptv', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling*rescalingI, debias=False, pol_solve=(1,1,1), pol_trans=True)
        
        self.func_msimple = EhtimFunctional(self.wrapper_msimple, domain)
        self.func_hw = EhtimFunctional(self.wrapper_hw, domain)
        self.func_ptv = EhtimFunctional(self.wrapper_ptv, domain)
        
        self.penalty_term = reg_term['msimple'] * self.func_msimple * Reshape(grid, domain)

        self.penalty_term2 = reg_term['hw'] * self.func_hw * Reshape(grid, domain)
        
        self.penalty_term3 = reg_term['ptv'] * self.func_ptv * Reshape(grid, domain)
        
        self.penalty_term = self.penalty_term * op
        self.penalty_term2 = self.penalty_term2 * op
        self.penalty_term3 = self.penalty_term3 * op
        
        if mode == 'pareto':
            self.ob8 = self.data_fidelity_term + self.penalty_term
            self.ob9 = self.data_fidelity_term + self.penalty_term2
            self.ob10 = self.data_fidelity_term + self.penalty_term3
            self.ob11 = self.data_fidelity_term
            
        elif mode == 'shapley':
            self.ob8 = self.penalty_term
            self.ob9 = self.penalty_term2
            self.ob10 = self.penalty_term3
            self.ob11 = self.data_fidelity_term
            
        else:
            print('mode not implemented')
            raise NotImplementedError
        
        ###Total Intensity###
        self.ob1 = self.ob1 * proj_I
        self.ob2 = self.ob2 * proj_I
        self.ob3 = self.ob3 * proj_I
        self.ob4 = self.ob4 * proj_I
        self.ob5 = self.ob5 * proj_I
        self.ob6 = self.ob6 * proj_I
        self.ob7 = self.ob7 * proj_I
        

class IQU:
    """
    A multi-objective problem.
    (This is actually a Python implementation of 2-dimensional ZDT-1 problem)

    USAGE: my_mo_problem()
    """
        
    def __init__(self, obs, prior, data_term, reg_term, rescaling, zbl, dim, num_cores=16, ttype='direct', mode='pareto'):
        self.obs = obs
        self.prior = prior
        self.data_term = data_term
        self.reg_term = reg_term
        self.rescaling = rescaling
        self.zbl = zbl
        self.dim = dim
        self.mode = mode
        
        self.num_cores = num_cores
        
    def setFit(self):
        self.fit = MyFunc(self.obs, self.prior, self.data_term, self.reg_term, rescaling=self.rescaling, zbl=self.zbl, mode=self.mode)
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
        return [self.fit.ob1(x),self.fit.ob2(x),self.fit.ob3(x),self.fit.ob4(x),self.fit.ob5(x), self.fit.ob6(x), self.fit.ob7(x), self.fit.ob8(x), self.fit.ob9(x), self.fit.ob10(x), self.fit.ob11(x)]
    
    def get_nobj(self):
        return 11

    def get_bounds(self):
        return (np.full((self.dim,),0.),np.full((self.dim,),1))


    def gradient(self, x):
        return np.concatenate(( self.fit.ob1.gradient(x), self.fit.ob2.gradient(x), self.fit.ob3.gradient(x), 
                              self.fit.ob4.gradient(x), self.fit.ob5.gradient(x), self.fit.ob6.gradient(x), self.fit.ob7.gradient(x), self.fit.ob8.gradient(x), self.fit.ob9.gradient(x), self.fit.ob10.gradient(x), self.fit.ob11.gradient(x) )) 
        #return estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        return "IQU"