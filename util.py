import numpy as np
import theano
import theano.tensor as T

def to_shared(param):
   return theano.shared(param, borrow=True)

def numpy_floatX(data):
   return np.asarray(data, dtype=theano.config.floatX)

