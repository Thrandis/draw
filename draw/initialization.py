"""Objects for encapsulating parameter initialization strategies."""
from abc import ABCMeta, abstractmethod
import numbers

import numpy
import theano
from six import add_metaclass

from blocks.initialization import NdarrayInitialization
from blocks.utils import repr_attrs


class Orthogonal(NdarrayInitialization):
    """Initialize a random orthogonal matrix.

    Parameters
    ----------
    scale : float
        Scale of the lines of the weight matrix (default=1.0)

    """
    def __init__(self, scale=1):
        self.scale = scale

    def generate(self, rng, shape):
        if len(shape) != 2:
            raise ValueError

        if shape[0] == shape[1]/4:
            print 'real ortho'
            l = [] 
            for i in range(4):
                a = rng.normal(0.0, 1.0, (shape[1]/4, shape[1]/4))
                u, _, v = numpy.linalg.svd(a, full_matrices=False)
                l.append(u[:shape[0], :])
            val = numpy.concatenate(l, axis=1).astype(theano.config.floatX)
        else:
            flat_shape = (shape[0], numpy.prod(shape[1:]))
            a = rng.normal(0.0, 1.0, flat_shape)
            u, _, v = numpy.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shp
            q = q.reshape(shape)
            val = q[:shape[0], :shape[1]].astype(theano.config.floatX)

        n = numpy.sqrt((val**2).sum(axis=0, keepdims=True))
        val /= n
        val *= self.scale
        return val


class NormalizedInitialization(NdarrayInitialization):
    """Initialize parameters with Glorot method.

    Notes
    -----
    For details see
    Understanding the difficulty of training deep feedforward neural networks,
    Glorot, Bengio, 2010

    """
    def generate(self, rng, shape):
        if len(shape) != 2:
            raise NotImplementedError
        else:
            input_size, output_size = shape
            high = numpy.sqrt(6) / numpy.sqrt(input_size + output_size)
            m = rng.uniform(-high, high, size=shape)
        return m.astype(theano.config.floatX)
