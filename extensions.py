import sys
import numpy
import theano
from theano import tensor as T
from theano import function
from cStringIO import StringIO

from blocks.extensions import SimpleExtension, Printing
from blocks.serialization import secure_dump


class PrintingTo(Printing):

    def __init__(self, path, **kwargs):
        super(PrintingTo, self).__init__(**kwargs)
        self.path = path
        with open(self.path, "w") as f:
            f.truncate(0)

    def do(self, *args, **kwargs):
        stdout, stringio = sys.stdout, StringIO()
        sys.stdout = stringio
        super(PrintingTo, self).do(*args, **kwargs)
        sys.stdout = stdout
        lines = stringio.getvalue().splitlines()
        with open(self.path, "a") as f:
            f.write("\n".join(lines))
            f.write("\n")


class ForceL2Norm(SimpleExtension):

    def __init__(self, variables, **kwargs):
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('after_batch', True)
        super(ForceL2Norm, self).__init__(**kwargs)
        self.variables = variables
        updates = []
        for variable in variables:
            norm = T.sqrt((variable**2).sum(axis=0, keepdims=True))  #TODO Check axis
            updates.append((variable, variable/norm))
        self.function = function([], [], updates=updates)

    def do(self, which_callback, *args):
        self.function()
