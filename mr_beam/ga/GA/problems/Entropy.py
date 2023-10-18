
from pygmo import *
import numpy as np

class Entropy:
    """
    A multi-objective problem.
    (This is actually a Python implementation of 2-dimensional ZDT-1 problem)

    USAGE: my_mo_problem()
    """
    def __init__(self,dim):
        self.dim = dim
        
    # Reimplement the virtual method that defines the objective function
    def fitness(self, x):
        x = np.abs(x)
        return [1/np.sum(np.log(x)*x),0]
    
    def get_nobj(self):
        return 2

    def get_bounds(self):
        return (np.full((self.dim,),-5.),np.full((self.dim,),10.))

    # Return function name
    def get_name(self):
        return "my function"
