import keras 
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape
from keras import Model
from tensorflow_addons.layers import MaxUnpooling2D
from tensorflow.nn import relu, sigmoid, max_pool_with_argmax
from keras.initializers import Constant

import numpy as np

class Encoder(Layer):
    def __init__(self, nr_channels=64, filter_size=3, ksize=2, strides=2, layers=2, layers_per_pool=1, dense=10):
        super(Encoder, self).__init__()
        
        self.nr_channels=nr_channels
        self.filter_size = filter_size
        self.ksize = ksize
        self.strides = strides
        self.layers = layers
        self.layers_per_pool = layers_per_pool
        
        self.masks = []
        self.conv = []
        
        for i in range(self.layers):
            for j in range(self.layers_per_pool):
                self.conv.append(Conv2D(self.nr_channels, (self.filter_size, self.filter_size), padding='same'))
        
        self.flatten = Flatten()        
        self.dense = Dense(dense)
        
    def call(self, x):
        self.masks = []
        
        for j in range(self.layers_per_pool):
            x = self.conv[j](x)
            x = relu(x)
        x, mask = max_pool_with_argmax(x, ksize=self.ksize, strides=self.strides, padding='SAME')
        self.masks.append(mask)
        
        for i in range(self.layers-1):
            for j in range(self.layers_per_pool):
                x = self.conv[(i+1)*self.layers_per_pool+j](x)
                x = relu(x)
            x, mask = max_pool_with_argmax(x, ksize=self.ksize, strides=self.strides, padding='SAME')
            self.masks.append(mask)
            
        x = self.flatten(x)
        x = self.dense(x)
        x = relu(x)
        
        return x, self.masks
    
class Decoder(Layer):
    def __init__(self, nr_channels=64, filter_size=3, ksize=2, strides=2, layers=2, layers_per_pool=1, shape=(32, 32)):
        super(Decoder, self).__init__()
        
        self.nr_channels=nr_channels
        self.filter_size = filter_size
        self.ksize = ksize
        self.strides = strides
        self.layers = layers
        self.layers_per_pool = layers_per_pool
        
        self.masks = []
        self.conv = []
        self.unpool = []
        
        for i in range(self.layers):
            for j in range(self.layers_per_pool):
                self.conv.append(Conv2D(self.nr_channels, (self.filter_size, self.filter_size), padding='same'))
        self.conv_final = Conv2D(1, (self.filter_size, self.filter_size), padding='same')
              
        for i in range(self.layers):
            self.unpool.append(MaxUnpooling2D())
            
        encoding_dim = ( shape[0]//(strides**layers), shape[1]//(strides**layers), nr_channels )
        
        self.reshape = Reshape(encoding_dim)
        self.dense = Dense(np.prod(encoding_dim))
            
    def call(self, x, masks):
        x = self.dense(x)
        x = relu(x)
        x = self.reshape(x)
        
        x = self.unpool[-1](x, masks[-1])
        
        for j in range(self.layers_per_pool):
            x = self.conv[0](x)
            x = relu(x)
        
        for i in range(self.layers-1):
            x = self.unpool[self.layers-i-2](x, masks[self.layers-i-2])
            for j in range(self.layers_per_pool):
                x = self.conv[(i+1)*self.layers_per_pool+j](x)
                x = relu(x)
                
        x = self.conv_final(x)
        x = sigmoid(x)
        
        return x
    
class BeamFinder(Model):
    def __init__(self, nr_channels=64, ksize=2, strides=2, layers=2, layers_per_pool=1, shape=(32, 32), filter_size=3, dense=10):
        super(BeamFinder, self).__init__()
        self.encoder = Encoder(nr_channels=nr_channels, filter_size=filter_size, ksize=ksize, strides=strides, layers=layers, layers_per_pool=1, dense=dense)
        self.decoder = Decoder(nr_channels=nr_channels, filter_size=filter_size, ksize=ksize, strides=strides, layers=layers, layers_per_pool=1, shape=shape)
        
    def call(self, inputs):
        x, masks = self.encoder(inputs)
        x = self.decoder(x, masks)
        
        return x






