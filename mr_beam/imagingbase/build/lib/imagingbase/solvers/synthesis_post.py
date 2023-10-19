import numpy as np
#import logging
#from regpy.operators import Identity

#class SynthesisImaging():
#    def __init__(self, data, solver, stoprule, dictionary, setting, evaluation = None):
#        self.data = data
#        self.solver = solver
#        self.stoprule = stoprule
#        self.dictionary = dictionary
#        self.setting = setting
#        if evaluation == None:
#            evaluation = Identity(self.setting.Hcodomain, self.setting.Hcodomain)
#        self.evaluation = evaluation
        
#    def run(self):
#        atoms, _ = self.solver.run(self.stoprule) 
#        reco = self.dictionary(atoms)
#        reco_data = self.evaluation(self.setting.op(reco))
#        rescale = np.sum(np.abs(self.data))/np.sum(np.abs(reco_data))
#        reco *= rescale
#        reco_data *= rescale
#        residual = self.data-reco_data
#        reco += self.setting.op.inverse(residual)
#        reco_data = self.setting.op(reco)
#        return [reco, reco_data]

#from scipy.optimize import curve_fit   
#from regpy.operators.msi import DOGDictionary 
#from regpy.discrs import Discretization

from imagingbase.solvers.utils import BuildMerger

#from copy import deepcopy

class Post():
    def __init__(self, solver):
        self.solver = solver
        Builder = BuildMerger(self.solver)
        self.merger = Builder._build_merger()
        
        self.reco = self.solver.reco.copy()
        
    def run(self):
        indices = self._compute_indices()
        
        self.reco += self.merger(indices)
        
        self.solver.set_reco(self.reco)
        
        return self.solver.finalize()
        
    def _compute_indices(self):
        indices = np.zeros((self.solver.solver.length, self.solver.solver.shape[0], self.solver.solver.shape[1]))
        for i in range(len(self.solver.cmap_list)):
            cmap = self.solver.cmap_list[i]
            for j in range(len(cmap)):
                scale = cmap[j][0]
                position = cmap[j][1]
                strength = cmap[j][2]
                gain = cmap[j][3]
                indices[scale, position[0], position[1]] += gain * strength / self.solver.solver.normalization[scale]
        return indices.flatten()