from __future__ import division

import numpy as np
import theano
import theano.tensor as T
import pdb
import pickle
from collections import OrderedDict



class AdamOptimizer:

    def __init__(self,params,learning_rate, sampleSize,b1=0.9, b2=0.9, batch_size=5, l2=0 ):


        self.lr = T.scalar("lr")

        self.params= params
        self.learning_rate = learning_rate
        self.b1=b1
        self.b2=b2
        self.batch_size = batch_size
        self.l2 = l2
        self.N = sampleSize

        self.m = OrderedDict()
        self.v = OrderedDict()

        for key, value in self.params.items():
            self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
            self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

    def get_updates(self, gradients, epoch,ascent=True):

        epsilon=1e-8
        updates = OrderedDict()
        updateTrace = OrderedDict()
        gamma = T.sqrt(1 - self.b2**epoch) / (1 - self.b1**epoch)

        values_iterable = zip(self.params.keys(), self.params.values(), gradients, 
                              self.m.values(), self.v.values())

        for name, parameter, gradient, m, v in values_iterable:
            new_m = self.b1 * m + (1. - self.b1) * gradient
            new_v = self.b2 * v + (1. - self.b2) * (gradient**2)

            updateTrace[parameter] = gamma * new_m / (T.sqrt(new_v) + epsilon)
            if ascent:
                updates[parameter] = parameter + self.lr * updateTrace[parameter]
            else:
                updates[parameter] = parameter - self.lr * updateTrace[parameter]

            if 'W' in name or 'gamma' in name:
                if ascent:
                    updates[parameter] += self.lr * self.l2 * T.mean(parameter)
                else:
                    updates[parameter] -= self.lr * self.l2 * T.mean(parameter)

            updates[m] = new_m
            updates[v] = new_v

        return [updates,updateTrace]


class RMSPROP:

    def __init__(self,params,learning_rate,batch_size,decay_rate=0.9,l2=0):


        self.lr = T.scalar("lr")
        self.batch_size = batch_size
        self.params= params
        self.learning_rate = learning_rate
        self.decay_rate=decay_rate
        self.l2=l2
        self.cache = OrderedDict()

        for key, value in self.params.items():
            self.cache[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='cache_' + key)

    def get_updates(self, gradients, epoch,ascent=True):

        epsilon=1e-8
        updates = OrderedDict()
        updateTrace = OrderedDict()
        values_iterable = zip(self.params.keys(), self.params.values(), gradients, 
                              self.cache.values())

        for name, parameter, gradient, cache in values_iterable:
            new_cache = self.decay_rate * cache + (1. - self.decay_rate) * (gradient**2)
            updateTrace[parameter] = gradient / (T.sqrt(new_cache) + epsilon)
            if ascent:
                updates[parameter] = parameter + self.lr  * updateTrace[parameter]
            else:
                updates[parameter] = parameter - self.lr  * updateTrace[parameter]
            # self.val = new_cache
            if 'W' in name or 'gamma' in name:
                if ascent:
                    updates[parameter] += self.lr * self.l2 * T.mean(parameter)
                else:
                    updates[parameter] -= self.lr * self.l2 * T.mean(parameter)
            updates[cache] = new_cache
        
      

        
        return [updates,updateTrace]
    
class GDA:
    """
    Gradient Descent Ascent
    """
    def __init__(self,params,learning_rate,batch_size,l2=0):


        self.lr = T.scalar("lr")
        self.batch_size = batch_size
        self.params= params
        self.learning_rate = learning_rate
        self.l2=l2

    def get_updates(self, gradients, epoch,ascent=True):

        epsilon=1e-8
        updates = OrderedDict()
        updateTrace = OrderedDict()
        values_iterable = zip(self.params.keys(), self.params.values(), gradients)

        for name, parameter, gradient in values_iterable:
            updateTrace[parameter] = gradient
            if ascent:
                updates[parameter] = parameter + self.lr  * updateTrace[parameter] 
            else:
                updates[parameter] = parameter - self.lr  * updateTrace[parameter] 
            
            if 'W' in name or 'gamma' in name:
                if ascent:
                    updates[parameter] += self.lr * self.l2 * T.mean(parameter)
                else:
                    updates[parameter] -= self.lr * self.l2 * T.mean(parameter)
        return [updates,updateTrace]    
