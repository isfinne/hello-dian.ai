
import numpy as np
from .modules import Module

def avoidZeros(element):
    return element if element != 0 else 1e-8
nonZeros = np.vectorize(avoidZeros)


# did i write correctly ?
class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        self.x = x
        self.f_sigmoid = lambda r: 1 / (1 + np.exp(-r))
        self.results = self.f_sigmoid(r= x)
        return self.results
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.
        derivative = self.f_sigmoid(self.x)*(1-self.f_sigmoid(self.x))
        dx = derivative * dy
        return dx
        # End of todo

class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.
        self.x = x
        return np.tanh(x)
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.
        derivative  = 1 - (np.tanh(self.x))**2
        dx = derivative * dy
        return dx
        # End of todo

class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        self.x = x
        self.results = np.maximum(1*x, 1e-2*x)
        return self.results
        # End of todos

    def backward(self, dy):
        # TODO Implement backward propogation
        # of ReLU function.
        
        derivative = np.array(self.x, copy= True)
        derivative[self.x > 0] = 1
        derivative[self.x <= 0 ] = 1e-2
        dx = derivative * dy
        return  dx 
        # End of todo

class Softmax(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of Softmax function.
        
        self.x = x
        #bias = np.max(x)
        #self.f_softmax = lambda r: np.exp(r)/np.sum(np.exp(r))
        C = np.tile(np.max(x, axis= 1).reshape(x.shape[0],1), (1,x.shape[1]))
        X_exp = np.exp(x-C)
        partition = np.tile(np.sum(X_exp, axis= 1).reshape(x.shape[0],1), (1,x.shape[1]))
        partition = nonZeros(partition)
        self.r = X_exp / partition
        #print(np.sum(self.r, axis= 1))
        return self.r
        # End of todo

    def backward(self, dy):
        derivative = np.diag(self.r) - np.dot(self.r.T, self.r)
        dx = derivative * dy
        return dx
        # Omitted.
        ...

class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...

class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.
        softmax = Softmax()
        self.probs = softmax.forward(probs)
        self.targets = targets
        
        #f_softmax = lambda r: np.exp(r)/np.sum(np.exp(r))
        #bias = np.max(self.probs)
        #y = f_softmax(self.probs - bias)
        #loss = -1 * np.log(y) * self.targets
        
        #self.loss = np.sum(-np.log(np.argmax(self.probs)) * self.targets)
        return self
        # End of todo

    def backward(self):
        # TODO Implement backward propogation
        # of softmax loss function.
        right = np.zeros_like(self.probs)
        for index, row in enumerate(right):
            p = self.targets[index]
            row[p] = 1
        
        return self.probs - right
        # End of todo

# how to write the backward part for only crossentropyloss function
class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):
        # TODO Calculate cross-entropy loss.
        self.probs = probs
        self.targets = targets
        X = probs
        partition = np.tile(np.sum(X, axis= 1).reshape(X.shape[0],1), (1,10))
        self.probs = X / partition
        
        #self.value = np.sum(-np.log(np.argmax(self.probs)) * self.targets)
        return self
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.
        right = np.zeros_like(self.probs)
        for index, row in enumerate(right):
            p = self.targets[index]
            row[p] = 1
        
        return self.probs - right
        # End of todo
