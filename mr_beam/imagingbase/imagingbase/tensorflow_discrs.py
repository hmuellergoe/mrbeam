from copy import copy
import numpy as np
from itertools import accumulate

from regpy.discrs import Discretization

from regpy import util, operators

import tensorflow as tf

class TensorflowSpace(Discretization):
    
    log = util.classlogger
    
    def __init__(self, shape, dtype=tf.float32):
        self.dtype = dtype
        """The discretization's dtype"""
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)
        self.shape = tf.TensorShape(shape)
        
    def zeros(self, dtype=None):
        return tf.zeros(self.shape, dtype=dtype or self.dtype)
    
    def ones(self, dtype=None):
        return tf.ones(self.shape, dtype=dtype or self.dtype)

    def empty(self, dtype=None):
        dtype = dtype or self.dtype
        numpy_array = np.empty(self.shape, dtype=dtype.as_numpy_dtype)
        return tf.convert_to_tensor(numpy_array, dtype=dtype)

    def iter_basis(self):
        elm = self.zeros().numpy()
        for idx in np.ndindex(tuple(self.shape)):
            elm[idx] = 1
            yield elm
            if self.is_complex:
                elm[idx] = 1j
                yield elm
            elm[idx] = 0
        elm = tf.convert_to_tensor(elm, dtype=self.dtype)

    def rand(self, rand=np.random.random_sample, dtype=None):
        dtype = dtype or self.dtype
        r = rand(tuple(self.shape))
#        if not np.can_cast(r.dtype, dtype.as_numpy_dtype):
#            raise ValueError(
#                'random generator {} can not produce values of dtype {}'.format(rand, dtype))
        if dtype.is_complex and not util.is_complex_dtype(r.dtype):
            c = np.empty(tuple(self.shape), dtype=dtype.as_numpy_dtype)
            c.real = r
            c.imag = rand(tuple(self.shape))
            return tf.convert_to_tensor(c, dtype=dtype)
        else:
            return tf.convert_to_tensor(np.asarray(r, dtype=dtype.as_numpy_dtype), dtype=dtype)

    @property
    def is_complex(self):
        """Boolean indicating whether the dtype is complex"""
        return self.dtype.is_complex

    @property
    def size(self):
        return np.prod(tuple(self.shape))

    @property
    def realsize(self):
        if self.is_complex:
            return 2 * np.prod(tuple(self.shape))
        else:
            return np.prod(tuple(self.shape))

    def __contains__(self, x):
        if x.shape != self.shape:
            return False
        elif x.dtype.is_complex:
            return self.is_complex
        elif not x.dtype.is_complex:
            return True
        else:
            return False

    def flatten(self, x):
        x_np = np.asarray(x)
        assert tuple(self.shape) == x_np.shape
        if self.is_complex:
            if x.dtype.is_complex:
                return tf.convert_to_tensor(util.complex2real(x_np).ravel())
            else:
                aux = np.asarray(self.empty())
                aux.real = x_np
                return tf.convert_to_tensor(util.complex2real(aux).ravel())
        elif x.dtype.is_complex:
            raise TypeError('Real discretization can not handle complex vectors')
        return tf.convert_to_tensor(x_np.ravel())

    def fromflat(self, x):
        x_np = np.asarray(x)
        assert util.is_real_dtype(x_np.dtype)
        if self.is_complex:
            return tf.convert_to_tensor(util.real2complex(x_np.reshape(self.shape + (2,))))
        else:
            return tf.convert_to_tensor(x_np.reshape(tuple(self.shape)))

    def complex_space(self):
        other = copy(self)
        other.dtype = tf.dtypes.as_dtype(np.result_type(1j, self.dtype.as_numpy_dtype))
        return other

    def real_space(self):
        other = copy(self)
        other.dtype = tf.dtypes.as_dtype(np.empty(0, dtype=self.dtype.as_numpy_dtype).real.dtype)
        return other
    
    def to_numpy(self, x):
        assert x in self
        return x.numpy()
    
    def from_numpy(self, x):
        toret = tf.convert_to_tensor(x)
        assert toret in self
        return toret

