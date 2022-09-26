import enum
from re import X
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn import Knn
import os, sys
dst = sys.argv[0]
selfPath = os.path.split(dst)[0] + '/'
def load_mnist(root= selfPath + 'mnist/'):
    # TODO Load the MNIST dataset
    fileList = os.listdir(root)
    dataSet = []
    for fileName in fileList:
        dataSet.append(np.fromfile(root + fileName, dtype= np.uint8))
        pass
    x_train = dataSet[0][16:].reshape(60000, 28, 28)
    y_train = dataSet[1][8:].reshape(60000)
    x_test = dataSet[2][16:].reshape(10000, 28, 28)
    y_test = dataSet[3][8:].reshape(10000)
    return x_train, y_train, x_test, y_test
    # End of todo


def main():
    #load_mnist()
    X_train, y_train, X_test, y_test = load_mnist()
    
    knn = Knn(k = 20)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #print(y_pred.shape)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()
