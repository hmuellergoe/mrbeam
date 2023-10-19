
from pygmo import *
import numpy as np

class Polynomial:
    """
    A multi-objective problem.
    (This is actually a Python implementation of 2-dimensional ZDT-1 problem)

    USAGE: my_mo_problem()
    """
    def __init__(self,dim):
        self.dim = dim
        
    # Reimplement the virtual method that defines the objective function
    def fitness(self, x):
        f0 = 5*(x[0]-0.1)**2 + (x[1]-0.1)**2
        f1 = (x[0]-0.9)**2 + 5*(x[1]-0.9)**2
        return [f0, f1]
    
    def get_nobj(self):
        return 2

    def get_bounds(self):
        return (np.full((self.dim,),-5.),np.full((self.dim,),10.))

    # Return function name
    def get_name(self):
        return "my function"
