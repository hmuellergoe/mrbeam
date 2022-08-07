import tensorflow as tf

from regpy.operators import Operator
from regpy.discrs import Discretization
from imagingbase.tensorflow_discrs import TensorflowSpace

class TensorflowBase():
    def __init__(self):
        return
    
    def tf_to_numpy(self, op):
        return TensorflowNumpyWrapper(op)
    
    def from_numpy(self, op):
        return NumpyTensorflowWrapper(op)
    
    def from_func(self, func, domain, codomain=None):
        return TensorflowOperator(func, domain, codomain)
    
        
class TensorflowOperator(Operator):
    def __init__(self, func, domain, codomain=None):        
        self.func = func
        codomain = codomain or domain
        super().__init__(domain, codomain)
        
    def _eval(self, x, differentiate=False):
        return self.func(x)
        
    def _derivative(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.func(x)            
        return tape.gradient(y, x)
    
    def to_numpy(self):
        return TensorflowNumpyWrapper(self)
        

class TensorflowNumpyWrapper(Operator):
#takes a tensorflow operator and wraps it to an operator based on numpy spaces
    def __init__(self, op):
        assert isinstance(op, Operator)
        self.op = op
        super().__init__(Discretization(op.domain.shape), Discretization(op.codomain.shape))
        
    def _eval(self, x, differentiate=False):
        return self.op.codomain.to_numpy(self.op(self.op.domain.from_numpy(x)))
    
    def _derivative(self, x):
        return self.op.codomain.to_numpy(self.op._derivative(self.op.domain.from_numpy(x)))
        
    def _adjoint(self, x):
        return self.op.domain.to_numpy(self.op.adjoint(self.op.codomain.from_numpy(x)))        


class NumpyTensorflowWrapper(Operator):
#takes a numpy operator and wraps it to an operator based on tensorflow spaces
    def __init__(self, op):
        assert isinstance(op, Operator)
        self.op = op
        super().__init__(TensorflowSpace(op.domain.shape), TensorflowSpace(op.codomain.shape))
        
    def _eval(self, x, differentiate=False):
        return self.codomain.from_numpy(self.op(self.domain.to_numpy(x)))
    
    def _derivative(self, x):
        return self.codomain.from_numpy(self.op._derivative(self.domain.to_numpy(x)))
        
    def _adjoint(self, x):
        return self.domain.from_numpy(self.op.adjoint(self.codomain.to_numpy(x)))
