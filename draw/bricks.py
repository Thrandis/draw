import copy
import numpy

from toolz import interleave
from picklable_itertools.extras import equizip

from theano import config, tensor
from blocks.bricks.base import application, lazy, Brick
from blocks.bricks.interfaces import Initializable, Feedforward, LinearLike
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks.sequences import FeedforwardSequence
from blocks.roles import add_role, WEIGHT, BIAS, INITIAL_STATE, PARAMETER
from blocks.utils import shared_floatx_nans, shared_floatx_zeros

WEIGHT_NORM = True 
print 'WEIGHT_NORM', WEIGHT_NORM

INITIAL_GAMMA = 1.0 
print 'INITIAL_GAMMA', INITIAL_GAMMA


class Linear(LinearLike, Feedforward):

    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim, norm=WEIGHT_NORM, initial_gamma=INITIAL_GAMMA,
                 **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = norm
        self.initial_gamma = initial_gamma

    def _allocate(self):
        W = shared_floatx_nans((self.input_dim, self.output_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if getattr(self, 'use_bias', True):
            b = shared_floatx_nans((self.output_dim,), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')
        if self.norm:
            self.gamma = shared_floatx_nans((self.output_dim,), name='gamma')
            add_role(self.gamma, PARAMETER)
            self.parameters.append(self.gamma)
            self.add_auxiliary_variable(self.gamma.norm(2), name='gamma_norm')

    def _initialize(self):
        if getattr(self, 'use_bias', True):
            self.biases_init.initialize(self.parameters[1], self.rng)
        self.weights_init.initialize(self.parameters[0], self.rng)
        if self.norm:
            if self.initial_gamma == 'auto':
                w = self.parameters[0].get_value()
                value = numpy.sqrt((w**2).sum(axis=0))
            else:
                value = self.initial_gamma * numpy.ones(self.output_dim,
                                                        dtype=config.floatX)
            self.gamma.set_value(value)
        print 'Linear'
        print numpy.sqrt((self.parameters[0].get_value()**2).sum(axis=0))

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        output = tensor.dot(input_, self.W)
        if self.norm:
            n = tensor.sqrt((self.W**2).sum(axis=0, keepdims=True))
            output = output * self.gamma.dimshuffle('x', 0) / n
        if getattr(self, 'use_bias', True):
            output += self.b
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return self.output_dim
        super(Linear, self).get_dim(name)


class LSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, norm=WEIGHT_NORM, initial_h_gamma=INITIAL_GAMMA, initial_c_gamma=None, 
                 compensate=True, **kwargs):
        self.dim = dim
        self.norm = norm
        self.initial_h_gamma = initial_h_gamma
        self.initial_c_gamma = initial_c_gamma
        self.compensate = compensate
        super(LSTM, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.b = shared_floatx_zeros((4*self.dim,),
                                     name='b')
        # Forget Gate bias init
        bval = self.b.get_value()
        bval[self.dim:2*self.dim] = 1
        self.b.set_value(bval)
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [self.W_state, self.b, self.initial_state_,
                           self.initial_cells]
        if self.norm:
            if self.initial_h_gamma is not None:
                self.h_gamma = shared_floatx_nans((4*self.dim,), name='h_gamma')
                add_role(self.h_gamma, PARAMETER)
                self.parameters.append(self.h_gamma)
            if self.initial_c_gamma is not None:
                self.c_gamma = shared_floatx_nans((self.dim,), name='c_gamma')
                add_role(self.c_gamma, PARAMETER)
                self.parameters.append(self.c_gamma)

    def _initialize(self):
        self.weights_init.initialize(self.parameters[0], self.rng)
        if self.norm:
            if self.initial_h_gamma is not None:
                vh = self.initial_h_gamma * numpy.ones(4*self.dim,
                                                       dtype=config.floatX)
                self.h_gamma.set_value(vh)
            if self.initial_c_gamma is not None:
                vc = self.initial_c_gamma * numpy.ones(self.dim,
                                                       dtype=config.floatX)
                self.c_gamma.set_value(vc)
        print 'LSTM Wh'
        print numpy.sqrt((self.parameters[0].get_value()**2).sum(axis=0))

    def norm_tanh(self, x, gamma):
        rnd = numpy.random.randn(10000000).astype(config.floatX)
        y = numpy.tanh(gamma*rnd)
        scale = numpy.sqrt(numpy.var(y))
        scale = scale.astype(config.floatX)
        return tensor.tanh(x) / scale

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        activation = tensor.dot(states, self.W_state)
        if self.norm:
            n = tensor.sqrt((self.W_state**2).sum(axis=0, keepdims=True))
            activation /= n
            if self.initial_h_gamma is not None:
                activation *= self.h_gamma.dimshuffle('x', 0)
        activation += inputs + self.b
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0))
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3))
        if self.norm and self.initial_h_gamma is not None and self.compensate:
            g = self.norm_tanh(slice_last(activation, 2), self.initial_h_gamma)
        else:
            g = tensor.tanh(slice_last(activation, 2))
        next_cells = forget_gate * cells + in_gate * g
        if self.norm and self.initial_c_gamma is not None and self.compensate:
            next_states = out_gate * self.norm_tanh(next_cells * self.c_gamma.dimshuffle('x', 0),
                                                    self.initial_c_gamma)
        else:
            next_states = out_gate * tensor.tanh(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]


class LNLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, initial_c_gamma=1.0, **kwargs):
        self.dim = dim
        self.initial_c_gamma = initial_c_gamma
        super(LNLSTM, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LNLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.b = shared_floatx_zeros((4*self.dim,),
                                     name='b')
        # Forget Gate bias init
        bval = self.b.get_value()
        bval[self.dim:2*self.dim] = 1
        self.b.set_value(bval)
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [self.W_state, self.b, self.initial_state_,
                           self.initial_cells]
        self.c_gamma = shared_floatx_nans((self.dim,), name='c_gamma')
        add_role(self.c_gamma, PARAMETER)
        self.parameters.append(self.c_gamma)
        self.c_beta = shared_floatx_zeros((self.dim,), name='c_beta')
        add_role(self.c_beta, PARAMETER)
        self.parameters.append(self.c_beta)

    def _initialize(self):
        self.weights_init.initialize(self.parameters[0], self.rng)
        vc = self.initial_c_gamma * numpy.ones(self.dim,
                                               dtype=config.floatX)
        self.c_gamma.set_value(vc)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        activation = tensor.dot(states, self.W_state)
        activation += inputs + self.b
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0))
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3))
        g = tensor.tanh(slice_last(activation, 2))
        next_cells = forget_gate * cells + in_gate * g
        # layer norm only on the cells
        m = next_cells.mean(axis=1, keepdims=True)
        v = next_cells.var(axis=1, keepdims=True)
        ln_next_cells = (next_cells - m)/tensor.sqrt(v + 1e-6)
        ln_next_cells *= self.c_gamma.dimshuffle('x', 0)
        ln_next_cells += self.c_beta.dimshuffle('x', 0)
        next_states = out_gate * tensor.tanh(ln_next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells


class MLP(FeedforwardSequence, Initializable):
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, prototype=None, norm=WEIGHT_NORM,
                 initial_gamma=INITIAL_GAMMA, **kwargs):
        self.activations = activations
        self.prototype = (Linear(norm=norm, initial_gamma=initial_gamma)
                          if prototype is None else prototype)
        self.linear_transformations = []
        for i in range(len(activations)):
            linear = copy.deepcopy(self.prototype)
            name = self.prototype.__class__.__name__.lower()
            linear.name = '{}_{}'.format(name, i)
            self.linear_transformations.append(linear)
        # Interleave the transformations and activations
        application_methods = []
        for entity in interleave([self.linear_transformations, activations]):
            if entity is None:
                continue
            if isinstance(entity, Brick):
                application_methods.append(entity.apply)
            else:
                application_methods.append(entity)
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(MLP, self).__init__(application_methods, **kwargs)

    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in \
                equizip(self.dims[:-1], self.dims[1:],
                        self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            if getattr(self, 'use_bias', None) is not None:
                layer.use_bias = self.use_bias
