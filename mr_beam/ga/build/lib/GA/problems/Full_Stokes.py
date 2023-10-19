import pygmo as pg
import ehtim as eh
import numpy as np

from imagingbase.ehtim_wrapper import EhtimFunctional, EhtimWrapper
from imagingbase.ehtim_wrapper_pol import EhtimWrapperPol
from imagingbase.polimaging import PolImager
from regpy.discrs import Discretization
from regpy.operators import Reshape, CoordinateMask, CoordinateProjection, Shifted, Multiplication

from joblib import Parallel, delayed

class MyFunc():
    def __init__(self, obs, prior, data_term, reg_term, mask_tot, ttype='direct', rescaling=0.02, rescalingV=0.002, zbl=1, dim=1, mode='pareto'):
        #rescaling = 1
        self.wrapper = EhtimWrapperPol(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='pvis', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False, pol_solve=(1,1,1), pol_trans=True)
        
        domain = Discretization(self.wrapper.xtuple.shape)
        grid = Discretization(np.prod(domain.shape))
        
        self.func_pvis = EhtimFunctional(self.wrapper, domain)
        self.data_fidelity_term = data_term['pvis'] * self.func_pvis
        self.data_fidelity_term = self.data_fidelity_term * Reshape(grid, domain)
        
        #Add Stokes I to array
        mask = np.zeros(grid.size, dtype='bool')
        mask[self.wrapper.xtuple.shape[1]:] = True
        op = CoordinateMask(self.data_fidelity_term.domain, mask)
        shift = self.wrapper.inittuple.flatten()-op(self.wrapper.inittuple.flatten())

        proj = CoordinateProjection(self.data_fidelity_term.domain, mask)
        op = proj.adjoint + shift
        
        mask_phase = np.zeros(grid.size, dtype='bool')
        mask_phase[2*self.wrapper.xtuple.shape[1]:] = True
        
        proj_phase = CoordinateProjection(self.data_fidelity_term.domain, mask_phase)
        op = proj_phase.adjoint * (2*np.pi*proj_phase) * proj.adjoint + proj.adjoint + (-1) * proj_phase.adjoint * proj_phase * proj.adjoint + shift
        
        self.data_fidelity_term = self.data_fidelity_term * op
        
        #Embed Stokes V in data term, has same size as grid, so we just reuse
        maskV = np.zeros(grid.size, dtype='bool')
        maskV[:2*self.wrapper.xtuple.shape[1]] = True
        opV = CoordinateProjection(grid,maskV)
        self.data_fidelity_term = self.data_fidelity_term * opV

        #Penalty terms
        prior_floor = prior.copy()
        self.floor = 1/(prior.xdim*prior.ydim)*np.max(prior.imvec) * 1/rescaling
        prior_floor.imvec += 0*self.floor
        
        self.wrapper_msimple = EhtimWrapperPol(obs.copy(), prior_floor.copy(), prior_floor.copy(), zbl,
                            d='msimple', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False, pol_solve=(1,1,1), pol_trans=True)

        self.wrapper_hw = EhtimWrapperPol(obs.copy(), prior_floor.copy(), prior_floor.copy(), zbl,
                            d='hw', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False, pol_solve=(1,1,1), pol_trans=True)
        
        self.wrapper_ptv = EhtimWrapperPol(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='ptv', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescaling, debias=False, pol_solve=(1,1,1), pol_trans=True)
        
        self.func_msimple = EhtimFunctional(self.wrapper_msimple, domain)
        #self.func_hw = EhtimFunctional(self.wrapper_hw, domain)
        self.func_ptv = EhtimFunctional(self.wrapper_ptv, domain)
        
        self.penalty_term = reg_term['msimple'] * self.func_msimple * Reshape(grid, domain)

        #self.penalty_term2 = reg_term['hw'] * self.func_hw * Reshape(grid, domain)
        
        self.penalty_term3 = reg_term['ptv'] * self.func_ptv * Reshape(grid, domain)
        
        #Add Stokes I to array
        mask = np.zeros(grid.size, dtype='bool')
        mask[self.wrapper_msimple.xtuple.shape[1]:] = True
        op_floor = CoordinateMask(self.penalty_term.domain, mask)
        shift = self.wrapper_msimple.inittuple.flatten()-op_floor(self.wrapper_msimple.inittuple.flatten())

        proj = CoordinateProjection(self.penalty_term.domain, mask)
        op_floor = proj.adjoint + shift
        
        mask_phase = np.zeros(grid.size, dtype='bool')
        mask_phase[2*self.wrapper_msimple.xtuple.shape[1]:] = True
        
        proj_phase = CoordinateProjection(self.penalty_term.domain, mask_phase)
        op_floor = proj_phase.adjoint * (2*np.pi*proj_phase) * proj.adjoint + proj.adjoint + (-1) * proj_phase.adjoint * proj_phase * proj.adjoint + shift
        
        self.penalty_term = self.penalty_term * op_floor
        #self.penalty_term2 = self.penalty_term2 * op_floor
        self.penalty_term3 = self.penalty_term3 * op
        
        #Embed penalty terms in Stokes V
        self.penalty_term = self.penalty_term * opV
        #self.penalty_term2 = self.penalty_term2 * opV
        self.penalty_term3 = self.penalty_term3 * opV

        #Stokes V data term
        self.wrapperV = EhtimWrapperPol(obs.copy(), prior.copy(), prior.copy(), zbl,
                            d='pvis', maxit=100, ttype=ttype, clipfloor=-100,
                            rescaling=rescalingV, debias=False, pol_solve=(1,1,1), pol_trans=False, stokesv=True)
        maskV_neg = np.zeros(grid.size, dtype=bool)
        maskV_neg[2*self.wrapper.xtuple.shape[1]:] = True
        opV_neg = CoordinateProjection(grid,maskV_neg)
        self.data_fidelity_term_V = EhtimFunctional(self.wrapperV, opV_neg.codomain) * Multiplication(opV_neg.codomain, self.wrapper.inittuple[0]) * Shifted(opV_neg, -0.5*opV_neg.codomain.ones())

        #Second penalty term
        self.func_hw = HWFunctional(self.wrapper, self.wrapperV, self.wrapper.inittuple[0]+self.floor, self.data_fidelity_term.domain)
        self.penalty_term2 = reg_term['hw'] * self.func_hw
        #self.penalty_term2 = self.penalty_term2 * op_floor

        if mode == 'pareto':
            self.ob1 = self.data_fidelity_term + self.penalty_term
            self.ob2 = self.data_fidelity_term + self.penalty_term2
            self.ob3 = self.data_fidelity_term + self.penalty_term3
            self.ob4 = self.data_fidelity_term
            self.ob5 = self.data_fidelity_term_V
            
        elif mode == 'shapley':
            self.ob1 = self.penalty_term
            self.ob2 = self.penalty_term2
            self.ob3 = self.penalty_term3
            self.ob4 = self.data_fidelity_term
            self.ob5 = self.data_fidelity_term_V
            
        else:
            print('mode not implemented')
            raise NotImplementedError
        
        mask_tot_qu = np.concatenate([mask_tot,np.ones(len(mask_tot),dtype=bool),mask_tot])
        self.proj_qu = CoordinateProjection(self.ob1.domain, mask_tot_qu)
        
        self.ob1 = self.ob1 * self.proj_qu.adjoint
        self.ob2 = self.ob2 * self.proj_qu.adjoint
        self.ob3 = self.ob3 * self.proj_qu.adjoint
        self.ob4 = self.ob4 * self.proj_qu.adjoint
        self.ob5 = self.ob5 * self.proj_qu.adjoint
        
    def update_obs(self, obs):
        #the functionals have the same instance, no copies, so changing wrapper changes all functionals
        self.wrapper.updateobs(obs)
        self.wrapper_msimple.updateobs(obs)
        self.wrapper_hw.updateobs(obs)
        self.wrapper_ptv.updateobs(obs) 
        self.wrapperV.updateobs(obs)

from regpy.functionals import Functional
class HWFunctional(Functional):
    def __init__(self, handler, handlerV, iimage, domain):
        self.handler = handler
        self.handlerV = handlerV
        self.iimage = iimage
        #assert self.handler.pol_trans
        super().__init__(domain)

    def _eval(self, poltuple):
        """Holdaway-Wardle polarimetric entropy
        """
        
        if self.handler.norm_init: 
            norm = self.handler.flux
        else: 
            norm = 1

        mimage = np.sqrt(self.handler.rescaling**2 * poltuple[:self.handler.inittuple.shape[1]]**2 + self.handlerV.rescaling**2 *((poltuple[2*self.handler.inittuple.shape[1]:]-0.5))**2)
        S = -np.sum(self.iimage * (((1+mimage)/2) * np.log((1+mimage)/2) + ((1-mimage)/2) * np.log((1-mimage)/2)))
        return S/norm
    
    def _gradient(self, poltuple):
        """Gradient of the Holdaway-Wardle polarimetric entropy
        """
        if self.handler.norm_init: 
            norm = self.handler.flux
        else: 
            norm = 1

        zeros =  np.zeros(len(self.iimage))
        mimage = np.sqrt(self.handler.rescaling**2 * poltuple[:self.handler.inittuple.shape[1]]**2 + self.handlerV.rescaling**2 *((poltuple[2*self.handler.inittuple.shape[1]:]-0.5))**2)

        gradchi = zeros
        if self.handler.pol_solve[1]!=0:
            gradm = -self.iimage * np.arctanh(mimage)
        else:
            gradm = zeros

        gradlm = self.handler.rescaling * gradm * poltuple[:self.handler.inittuple.shape[1]] / mimage
        gradcm = self.handlerV.rescaling *gradm * (poltuple[2*self.handler.inittuple.shape[1]:] - 0.5)/ mimage
        out = (gradlm, gradchi, gradcm)

        return np.array(out)/norm


class FullStokes:
    """
    A multi-objective problem.
    (This is actually a Python implementation of 2-dimensional ZDT-1 problem)

    USAGE: my_mo_problem()
    """
        
    def __init__(self, obs_List, prior, data_term, reg_term, rescaling, zbl, dim, num_cores=16, ttype='direct', pcut=0, mode='pareto', rescalingV=None):
        self.obs_List = obs_List
        self.prior = prior
        self.data_term = data_term
        self.reg_term = reg_term
        self.rescaling = rescaling
        self.rescalingV = rescalingV or rescaling
        self.zbl = zbl
        if pcut == 0:
            self.dim = dim
            self.mask = np.ones(dim//3, dtype=bool)
        else:
            thresh = pcut*np.max(prior.imvec)
            self.mask = np.zeros(dim//3, dtype=bool)
            for i in range(len(self.mask)):
                self.mask[i] = (prior.imvec[i] > thresh)
            self.dim = 2*np.sum(self.mask)+dim//3
        self.mode = mode

        self.num_cores = num_cores
        
    def setFit(self):
        self.fit = MyFunc(self.obs_List, self.prior, self.data_term, self.reg_term, self.mask, rescaling=self.rescaling, rescalingV=self.rescalingV, zbl=self.zbl, mode=self.mode)
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
        #return [self.fit.ob3(x),self.fit.ob4(x)]
        return [self.fit.ob1(x),self.fit.ob2(x),self.fit.ob3(x),self.fit.ob4(x), self.fit.ob5(x)]
    
    def get_nobj(self):
        return 5
    
    def get_bounds(self):
        return (np.full((self.dim,),0.),np.full((self.dim,),1))


    def gradient(self, x):
        #return np.concatenate((self.fit.ob3.gradient(x), self.fit.ob4.gradient(x)))
        return np.concatenate(( self.fit.ob1.gradient(x), self.fit.ob2.gradient(x), self.fit.ob3.gradient(x), self.fit.ob4.gradient(x),self.fit.ob5.gradient(x))) 
        #return estimate_gradient_h(lambda x: self.fitness(x), x)

    # Return function name
    def get_name(self):
        return "simple polynomial example"






























