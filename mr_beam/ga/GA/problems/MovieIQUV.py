import pygmo as pg
import ehtim as eh
import numpy as np

from imagingbase.ehtim_wrapper import EhtimFunctional, EmptyFunctional
from imagingbase.ehtim_wrapper import EhtimWrapper
from imagingbase.ngMEM_functional import TemporalEntropy
from regpy.discrs import Discretization
from regpy.operators import Reshape
from regpy.functionals import FunctionalProductSpace

from joblib import Parallel, delayed

from GA.problems.IQUV import MyFunc as IQUVFunc
from regpy.operators import Reshape, CoordinateMask, CoordinateProjection

class MyFunc():
    def __init__(self, obs_List, prior, data_term, reg_term, zbls, ttype='direct', rescaling=0.02, rescalingV=0.002, dim=1, C=1, tau=1, mode='pareto'):
        self.nr_of_frames = len(obs_List)
        
        size = len(prior.imvec)
        grid = Discretization(4*size)
        
        obs = obs_List[0]
        
        ob1s = []
        ob2s = []
        ob3s = []
        ob4s = []        
        ob5s = []        
        ob6s = []        
        ob7s = []        
        ob8s = []        
        ob9s = []        
        ob10s = []        
        ob11s = []        
        ob12s = []        
        
        for i in range(self.nr_of_frames):
            func = IQUVFunc(obs_List[i], prior, data_term, reg_term, ttype=ttype, rescaling=rescaling, rescalingV=rescalingV, zbl=zbls[i], dim=dim, mode=mode)
            
            ob1s.append(func.ob1)
            ob2s.append(func.ob2)
            ob3s.append(func.ob3)            
            ob4s.append(func.ob4)
            ob5s.append(func.ob5)
            ob6s.append(func.ob6)
            ob7s.append(func.ob7)
            ob8s.append(func.ob8)
            ob9s.append(func.ob9)
            ob10s.append(func.ob10)
            ob11s.append(func.ob11)
            ob12s.append(func.ob12)

        self.ob1 = FunctionalProductSpace(ob1s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))
        self.ob2 = FunctionalProductSpace(ob2s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob3 = FunctionalProductSpace(ob3s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob4 = FunctionalProductSpace(ob4s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob5 = FunctionalProductSpace(ob5s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob6 = FunctionalProductSpace(ob6s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob7 = FunctionalProductSpace(ob7s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob8 = FunctionalProductSpace(ob8s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob9 = FunctionalProductSpace(ob9s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob10 = FunctionalProductSpace(ob10s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob11 = FunctionalProductSpace(ob11s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))        
        self.ob12 = FunctionalProductSpace(ob12s, grid**self.nr_of_frames, np.ones(self.nr_of_frames))             

        domain = Discretization(prior.imarr().shape)
        self.func_ngmem = TemporalEntropy(domain, self.nr_of_frames, C, tau)

        maskIs = np.zeros((self.nr_of_frames, 4*size), dtype=bool)
        for i in range(maskIs):
            maskIs[i][0:size] = True
        projI = CoordinateProjection(self.ob1.domain, maskIs.flatten())
        self.ob13 = self.func_ngmem * projI


