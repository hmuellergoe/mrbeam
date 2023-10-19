import pygmo as pg
import ehtim as eh
import numpy as np

from imagingbase.ehtim_wrapper import EhtimFunctional
from imagingbase.ehtim_wrapper import EhtimWrapper
from imagingbase.ngMEM_functional import TemporalEntropy
from regpy.discrs import Discretization
from regpy.operators import Reshape
from regpy.functionals import FunctionalProductSpace

from joblib import Parallel, delayed

class MyFunc():
    def __init__(self, obs_List, prior, data_term, reg_term, zbls, ttype='direct', rescaling=0.02, dim=1, C=1, tau=1, mode='pareto'):
        self.nr_of_frames = len(obs_List)
        
        domain = Discretization(prior.imarr().shape)
        grid = Discretization(np.prod(domain.shape))
        
        func_vis = []
        func_cph = []
        func_logcamp = []
        func_amp = []
        
        for i in range(self.nr_of_frames):
            wrapper = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                d='vis', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
    
            wrapper_cph = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                    d='vis', maxit=100, ttype=ttype, clipfloor=-100,
                                    rescaling=rescaling, debias=False)
            
            wrapper_logcamp = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                    d='vis', maxit=100, ttype=ttype, clipfloor=-100,
                                    rescaling=rescaling, debias=False)
            
            wrapper_amp = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                    d='vis', maxit=100, ttype=ttype, clipfloor=-100,
                                    rescaling=rescaling, debias=False)
        
            func_vis.append(EhtimFunctional(wrapper, domain))
            func_cph.append(EhtimFunctional(wrapper_cph, domain))
            func_logcamp.append(EhtimFunctional(wrapper_logcamp, domain))
            func_amp.append(EhtimFunctional(wrapper_amp, domain))
            
        self.func_vis = FunctionalProductSpace(func_vis, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_amp = FunctionalProductSpace(func_amp, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_cph = FunctionalProductSpace(func_cph, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_logcamp = FunctionalProductSpace(func_logcamp, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        
        self.data_fidelity_term = data_term['vis'] * self.func_vis \
                                    + data_term['amp'] * self.func_amp \
                                    + data_term['cphase'] * self.func_cph \
                                    + data_term['logcamp'] * self.func_logcamp
                                    
        self.data_fidelity_term = self.data_fidelity_term
        
        func_l1 = []
        func_simple = []
        func_tv = []
        func_tvs = []
        func_l2 = []
        func_flux = []
        
        for i in range(self.nr_of_frames):
        
            wrapper_l1 = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                d='l1w', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
    
            wrapper_simple = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                d='simple', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
            
            wrapper_tv = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                d='tv', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
            
            wrapper_tvs = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                d='tv2', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
                    
            wrapper_l2 = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                d='lA', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
            
            wrapper_flux = EhtimWrapper(obs_List[i], prior.copy(), prior.copy(), zbls[i],
                                d='flux', maxit=100, ttype=ttype, clipfloor=-100,
                                rescaling=rescaling, debias=False)
        
            func_l1.append(EhtimFunctional(wrapper_l1, domain))
            func_simple.append(EhtimFunctional(wrapper_simple, domain))
            func_tv.append(EhtimFunctional(wrapper_tv, domain))
            func_tvs.append(EhtimFunctional(wrapper_tvs, domain))
            func_l2.append(EhtimFunctional(wrapper_l2, domain))
            func_flux.append(EhtimFunctional(wrapper_flux, domain))
            
        self.func_l1 = FunctionalProductSpace(func_l1, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_simple = FunctionalProductSpace(func_simple, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_tv = FunctionalProductSpace(func_tv, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_tvs = FunctionalProductSpace(func_tvs, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_l2 = FunctionalProductSpace(func_l2, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_flux = FunctionalProductSpace(func_flux, domain**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.func_ngmem = TemporalEntropy(domain, self.nr_of_frames, C, tau)
        
        self.penalty_term = reg_term['l1w'] * self.func_l1
                            
        self.penalty_term2 = reg_term['tv'] * self.func_tv
        
        self.penalty_term3 = reg_term['tv2'] * self.func_tvs
        
        self.penalty_term4 = reg_term['lA'] * self.func_l2
        
        self.penalty_term5 = reg_term['flux'] * self.func_flux
        
        self.penalty_term6 = reg_term['simple'] * self.func_flux
        
        self.penalty_term7 = reg_term['ngmem'] * self.func_ngmem
        
        if mode == 'pareto':
            self.ob1 = self.data_fidelity_term + self.penalty_term
            self.ob2 = self.data_fidelity_term + self.penalty_term2
            self.ob3 = self.data_fidelity_term + self.penalty_term3
            self.ob4 = self.data_fidelity_term + self.penalty_term4
            self.ob5 = self.data_fidelity_term + self.penalty_term5
            self.ob6 = self.data_fidelity_term + self.penalty_term6
            self.ob7 = self.data_fidelity_term + self.penalty_term7
            self.ob8 = self.data_fidelity_term
            
        elif mode == 'shapley':
            self.ob1 = self.penalty_term
            self.ob2 = self.penalty_term2
            self.ob3 = self.penalty_term3
            self.ob4 = self.penalty_term4
            self.ob5 = self.penalty_term5
            self.ob6 = self.penalty_term6
            self.ob7 = self.penalty_term7
            self.ob8 = self.data_fidelity_term
            
        else:
            print('mode not implemented')
            raise NotImplementedError

class Movie:
    """
    A multi-objective problem.
    (This is actually a Python implementation of 2-dimensional ZDT-1 problem)

    USAGE: my_mo_problem()
    """
        
    def __init__(self, obs_List, prior, data_term, reg_term, rescaling, zbls, dim, num_cores=16, ttype='direct', C=1, tau=1, mode='pareto'):
        self.obs_List = obs_List
        self.prior = prior
        self.data_term = data_term
        self.reg_term = reg_term
        self.rescaling = rescaling
        self.zbls = zbls
        self.dim = dim
        self.C = 1
        self.tau = 1
        self.mode = mode
        
        self.num_cores = num_cores
        
    def setFit(self):
        self.fit = MyFunc(self.obs_List, self.prior, self.data_term, self.reg_term, rescaling=self.rescaling, zbls=self.zbls, C=self.C, tau=self.tau, mode=self.mode)
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
        return [self.fit.ob1(x),self.fit.ob2(x),self.fit.ob3(x),self.fit.ob4(x),self.fit.ob5(x), self.fit.ob6(x), self.fit.ob7(x), self.fit.ob8(x)]
    
    def get_nobj(self):
        return 8

    def get_bounds(self):
        return (np.full((self.dim,),0.),np.full((self.dim,),1))


    def gradient(self, x):
        return np.concatenate(( self.fit.ob1.gradient(x), self.fit.ob2.gradient(x), self.fit.ob3.gradient(x), 
                              self.fit.ob4.gradient(x), self.fit.ob5.gradient(x), self.fit.ob6.gradient(x), self.fit.ob7.gradient(x) , self.fit.ob8.gradient(x) )) 
        #return estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        return "simple polynomial example"
