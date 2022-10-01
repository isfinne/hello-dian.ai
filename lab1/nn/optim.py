from .tensor import Tensor
from .modules import Module
import numpy as np


class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):
        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.
        try:
            varsDict = vars(module)
            value = varsDict.values()
        except TypeError:
            # type 'list' do not have attribute '__dict__'
            value = module
        for attr in value:
            if isinstance(attr, Module) or isinstance(attr, list):
                # recursively call the method
                self._step_module(attr)
            elif isinstance(attr, Tensor):
                self._update_weight(attr)
        pass
        # End of todo
    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum

    def _update_weight(self, tensor):
        # TODO Update the weight of tensor
        # in SGD manner.
        try:
            tensor.grad
        except AttributeError:
           return
        try:
            tensor.v
        except AttributeError:
            tensor.v = np.zeros_like(tensor.grad)
        for i in range(8):
            tensor.v = self.momentum * tensor.v - self.lr * tensor.grad
        tensor += tensor.v 
        # End of todo

class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.

        ...

        # End of todo

    def _update_weight(self, tensor):

        # TODO Update the weight of
        # tensor in Adam manner.

        ...

        # End of todo
