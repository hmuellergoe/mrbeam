from imagingbase.regpy_functionals import Functional
from regpy.operators import Operator
import numpy as np
import tensorflow as tf

#use numpy arrays as inputs, then the correct dtypes appear

class NNL2Functional(Functional):
    def __init__(self, model, domain, loss='L2'):
        self.model = model
        if loss=='L2':
            self.loss = tf.nn.l2_loss
        else:
            print('Loss not implemented, use L2 instead')
            self.loss = tf.nn.l2_loss
        super().__init__(domain)

    def _eval(self, x):
        return float(self.loss(self.model(tf.convert_to_tensor(x))).numpy())
#        return float(np.linalg.norm(self.model(x).numpy()))

    def _gradient(self, x):
#        x = tf.cast(x, tf.float32)
        
#        with tf.GradientTape() as tape:
#            tape.watch(x)
#            y = self.model(x)
                  
#        jacobian = tape.batch_jacobian(y, x)
        
#        grad = jacobian.numpy().reshape((np.prod(self.domain.shape), np.prod(self.domain.shape))) @ (2*y.numpy().flatten())
                
#        return grad.reshape(self.domain.shape)
    
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.loss(self.model(x))
        grad = tape.gradient(y, x)
        
        return grad.numpy().reshape(self.domain.shape)

    def _proximal(self, x, tau):
        return NotImplementedError

from regpy.operators import Operator
class Normalization(Operator):
    def __init__(self, domain):
        self.norm_factor = 1
        super().__init__(domain, domain, linear=True)
        
    def _eval(self, x):
        self.norm_factor = np.max(x)
        return x / self.norm_factor
    
    def _adjoint(self, x):
        return self._eval(x)
    
    @property
    def inverse(self):
        return self.norm_factor * self.domain.identity
    
class NormalizedFunctional(Functional):
    def __init__(self, func):
        self.norm_op = Normalization(func.domain)
        self.func = func * self.norm_op
        super().__init__(self.func.domain)
        
    def _eval(self, x):
        toret = self.func(x)
        return self.norm_op.norm_factor * toret
    
    def _gradient(self, x):
        toret = self.func.gradient(x)
        return self.norm_op.norm_factor * toret
    
    def _proximal(self, x, tau):
        return NotImplementedError
        
