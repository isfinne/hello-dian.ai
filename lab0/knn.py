import numpy as np
from tqdm import tqdm


class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        # raise NotImplementedError
        ...

        # End of todo
