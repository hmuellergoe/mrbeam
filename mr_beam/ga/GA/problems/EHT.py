import pygmo as pg
import ehtim as eh
import numpy as np

from imagingbase.ehtim_wrapper import EhtimFunctional
from imagingbase.ehtim_wrapper import EhtimWrapper, EmptyFunctional
from regpy.discrs import Discretization
from regpy.operators import Reshape

from joblib import Parallel, delayed

class MyFunc():
    def __init__(self, obs, prior, data_term, reg_term, ttype='direct', rescaling=0.02, zbl=1, dim=1, mode='pareto'):
        
        domain = Discretization(prior.imarr().shape)
        grid = Discretization(np.prod(domain.shape))
        
        if data_term["vis"] == 0:
            self.func_vis = EmptyFunctional(domain)
        else:                
            self.wrapper = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='vis', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False)
            
            self.func_vis = EhtimFunctional(self.wrapper, domain)
            
        if data_term["cphase"] == 0:
            self.func_cph = EmptyFunctional(domain)
        else:                
            self.wrapper_cph = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                                d='cphase', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False, maxset=True)
            
            self.func_cph = EhtimFunctional(self.wrapper_cph, domain)
            
        if data_term["logcamp"] == 0:
            self.func_logcamp = EmptyFunctional(domain)
        else:                
            self.wrapper_logcamp = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                                d='logcamp', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False, maxset=True)
            
            self.func_logcamp = EhtimFunctional(self.wrapper_logcamp, domain)
            
        if data_term["amp"] == 0:
            self.func_amp = EmptyFunctional(domain)
        else:                
            self.wrapper_amp = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                                d='amp', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
            
            self.func_amp = EhtimFunctional(self.wrapper_amp, domain)
        
        self.data_fidelity_term = data_term['vis'] * self.func_vis \
                                    + data_term['amp'] * self.func_amp \
                                    + data_term['cphase'] * self.func_cph \
                                    + data_term['logcamp'] * self.func_logcamp
                                    
        self.data_fidelity_term = self.data_fidelity_term * Reshape(grid, domain)
        
        self.wrapper_l1 = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='l1w', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False)

        self.wrapper_simple = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='simple', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False)
        
        self.wrapper_tv = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='tv', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False, epsilon_tv=0.001)
        
        self.wrapper_tvs = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='tv2', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False)
                
        self.wrapper_l2 = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='lA', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False)
        
        self.wrapper_flux = EhtimWrapper(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='flux', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False)
        
        self.func_l1 = EhtimFunctional(self.wrapper_l1, domain)
        self.func_simple = EhtimFunctional(self.wrapper_simple, domain)
        self.func_tv = EhtimFunctional(self.wrapper_tv, domain)
        self.func_tvs = EhtimFunctional(self.wrapper_tvs, domain)
        self.func_l2 = EhtimFunctional(self.wrapper_l2, domain)
        self.func_flux = EhtimFunctional(self.wrapper_flux, domain)
        
        self.penalty_term = reg_term['l1w'] * self.func_l1
                            
        self.penalty_term = self.penalty_term * Reshape(grid, domain)

        self.penalty_term2 = reg_term['tv'] * self.func_tv
                            
        self.penalty_term2 = self.penalty_term2 * Reshape(grid, domain)
        
        self.penalty_term3 = reg_term['tv2'] * self.func_tvs
                            
        self.penalty_term3 = self.penalty_term3 * Reshape(grid, domain)
        
        self.penalty_term4 = reg_term['lA'] * self.func_l2
                            
        self.penalty_term4 = self.penalty_term4 * Reshape(grid, domain)
        
        self.penalty_term5 = reg_term['flux'] * self.func_flux
                            
        self.penalty_term5 = self.penalty_term5 * Reshape(grid, domain)
        
        self.penalty_term6 = reg_term['simple'] * self.func_flux
                            
        self.penalty_term6 = self.penalty_term6 * Reshape(grid, domain)
        
        if mode == 'pareto':
            self.ob1 = self.data_fidelity_term + self.penalty_term
            self.ob2 = self.data_fidelity_term + self.penalty_term2
            self.ob3 = self.data_fidelity_term + self.penalty_term3
            self.ob4 = self.data_fidelity_term + self.penalty_term4
            self.ob5 = self.data_fidelity_term + self.penalty_term5
            self.ob6 = self.data_fidelity_term + self.penalty_term6
            self.ob7 = self.data_fidelity_term
            
        elif mode == 'shapley':
            self.ob1 = self.penalty_term
            self.ob2 = self.penalty_term2
            self.ob3 = self.penalty_term3
            self.ob4 = self.penalty_term4
            self.ob5 = self.penalty_term5
            self.ob6 = self.penalty_term6
            self.ob7 = self.data_fidelity_term
            
        else:
            print('mode not implemented')
            raise NotImplementedError
            
    def update_obs(self, obs):
        #the functionals have the same instance, no copies, so changing wrapper changes all functionals
        try:
            self.wrapper.updateobs(obs)
        except:
            pass
        try:
            self.wrapper_cph.updateobs(obs)
        except:
            pass
        try:
            self.wrapper_logcamp.updateobs(obs)
        except:
            pass
        try:
            self.wrapper_amp.updateobs(obs)
        except:
            pass
        
        self.wrapper_l1.updateobs(obs)
        self.wrapper_simple.updateobs(obs)
        self.wrapper_tv.updateobs(obs)
        self.wrapper_tvs.updateobs(obs)
        self.wrapper_l2.updateobs(obs)
        self.wrapper_flux.updateobs(obs)
           

class EHT:
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
        self.ttype=ttype
        
    def setFit(self):
        self.fit = MyFunc(self.obs, self.prior, self.data_term, self.reg_term, ttype=self.ttype, rescaling=self.rescaling, zbl=self.zbl, mode=self.mode)
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
        return [self.fit.ob1(x),self.fit.ob2(x),self.fit.ob3(x),self.fit.ob4(x),self.fit.ob5(x), self.fit.ob6(x), self.fit.ob7(x)]
    
    def get_nobj(self):
        return 7

    def get_bounds(self):
        return (np.full((self.dim,),0.),np.full((self.dim,),1))


    def gradient(self, x):
        return np.concatenate(( self.fit.ob1.gradient(x), self.fit.ob2.gradient(x), self.fit.ob3.gradient(x), 
                              self.fit.ob4.gradient(x), self.fit.ob5.gradient(x), self.fit.ob6.gradient(x), self.fit.ob7.gradient(x) )) 
        #return estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        return "EHT"
