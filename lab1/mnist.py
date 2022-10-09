import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import nn
import nn.functional as F

n_features = 28 * 28
n_classes = 10
n_epochs = 10
bs = 1000
# lr = 2e-8 for SGD
# lr = 2e-2 for Adam
lr = 2e-2
lengths = (n_features, 512, n_classes)

class Layer(nn.Module):
    def __init__(self, L_in: int, L_out: int, f_activate: nn.Module= None,
                 weights: np.ndarray= None, bias: np.ndarray= None):
        super().__init__()
        self.L_in = L_in
        self.L_out = L_out
        self.activation = f_activate 
        self.w = np.concatenate((weights, bias)) if weights is not None else nn.tensor.tensor((L_in + 1, L_out))
        self.w.grad = np.zeros_like(self.w)
        pass
    
    def forward(self, x: np.ndarray):
        self.x = x
        self.length = x.shape[0]
        linearResults = np.dot(x, self.w[1:]) + np.tile(self.w[0], self.length).reshape(-1, self.L_out) 
        try:
            activation_output = self.activation.forward(linearResults)
        except AttributeError:
            activation_output = linearResults 
        return activation_output
        
    def backward(self, dy: np.ndarray):
        if not isinstance(self.activation, nn.Module):
            #df = super().backward(dy)
            pass
        #else: df = self.activation.backward(dy)
        else: df = dy
        # linear part output
        derivative = self.w[1:].T
        dx = np.dot(df, derivative)
        
        # caculate the weights and bias
        weight_grad = np.dot(self.x.T, df)/self.length
        bias_grad = (self.w[0] * np.sum(df, axis= 0)).reshape(1,self.L_out)/self.length #self.length
        self.w.grad = np.concatenate((bias_grad, weight_grad))
        
        return dx
    pass

class Model(nn.Module):

    # TODO Design the classifier.
    def __init__(self, lengths):
        super().__init__()
        self.lengths = lengths
        self._layers = []
        self.forwardOutput = None
        
    def addLayer(self, layer: Layer):
        self._layers.append(layer)
    
    def forward(self, X):
        for layer in self._layers:
            X = layer.forward(X)
        forwardOutput = X
        return forwardOutput    
    
    def backward(self, dy: np.ndarray):
        df = dy
        for i in reversed(range(len(self._layers))): # 反向循环
            layer = self._layers[i]
            df = layer.backward(df)
        pass
    

    # End of todo


def load_mnist(mode='train', n_samples=None, flatten=True):
    images = './mnist/train-images.idx3-ubyte' if mode == 'train' else './mnist/zt10k-images.idx3-ubyte'
    labels = './mnist/train-labels.idx1-ubyte' if mode == 'train' else './mnist/zt10k-labels.idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape(
        (length, 28, 28)).astype(np.int32)
    if flatten:
        X = X.reshape(length, -1)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape(
        (length)).astype(np.int32)
    return (X[:n_samples] if n_samples is not None else X,
            y[:n_samples] if n_samples is not None else y)


def vis_demo(model):
    X, y = load_mnist('test', 20)
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    fig = plt.subplots(nrows=4, ncols=5, sharex='all',
                       sharey='all')[1].flatten()
    for i in range(20):
        img = X[i].reshape(28, 28)
        fig[i].set_title(preds[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("vis.png")
    plt.show()


def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test'))
    model = Model(lengths)
    optimizer = nn.optim.Adam(model, lr=lr)
    #optimizer = nn.optim.SGD(model, lr=lr)
    criterion = F.CrossEntropyLoss(n_classes= n_classes)
    
    # Add different layers to the network.
    #model.addLayer(Layer(784, 256, F.ReLU()))
    #model.addLayer(Layer(256, 128, F.ReLU()))
    #model.addLayer(Layer(128, 64, F.ReLU()))
    #model.addLayer(Layer(64, 32, F.ReLU()))
    model.addLayer(Layer(784, 10, F.ReLU()))
    
    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4 / bs)
        bar.set_description(f'epoch  {i:2}')
        for X, y in bar:
            probs = model.forward(X)
            loss = criterion(probs, y)
            model.backward(loss.backward())
            optimizer.step()
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y) / len(y) * 100:.1f}'
                                ' loss={loss.value:.3f}')

        for X, y in testloader:
            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            print(f' test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

    vis_demo(model)


if __name__ == '__main__':
    main()
