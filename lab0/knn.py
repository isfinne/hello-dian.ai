import numpy as np
import torch
from tqdm import tqdm
import operator


def PCA(inX, K = 50):
    #C, U, Z, centrX = [],[],[],[]
    samplesNum = inX.shape[0]
    dataSetSize = inX.size
    #把m*N1*N1图像矩阵重排为m*(N1^2)的序列
    originMatrix = inX.reshape(samplesNum, int(dataSetSize/samplesNum)).T
    
    #centralMatrix
    mean_row = originMatrix.mean(axis= 1)  # 计算完之后array的长度等于列数
    mean_rowMatrix = mean_row.reshape(mean_row.shape[0], 1)
    mean_matrix = np.tile(mean_rowMatrix, (samplesNum))
    
    #print(mean_matrix.shape)
    #print(originMatrix.shape)
    centralMatrix = originMatrix - mean_matrix
    #print(len(originMatrix))
    
    covMatrix = np.dot(centralMatrix, centralMatrix.T)/samplesNum
    #print(covMatrix.shape)
    
    char_value, char_vector = np.linalg.eig(covMatrix)
    charValueIndexList = np.argsort(-1*char_value)
    
    vectorList = [char_vector[:,charValueIndexList[i]] for i in range(K)]
    
    PMatrix = np.reshape(vectorList,(K, char_value.shape[0]))
    #print(char_vector[:,charValueIndexList[0]].reshape(1,784).shape)
    resultMatrix = np.dot(PMatrix, originMatrix).T
    #print(resultMatrix.shape)
    #print(resultMatrix.shape)
    return resultMatrix
    pass

def judge(value):
    return 0 if value == 0 else 255
judgeZero = np.vectorize(judge)

class Knn(object):

    def __init__(self, k=30):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.Size = 60000
    
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
        # Totally 60000 groups of x_data, each data group is a 28x28 matrix and has a int label between 0-9
        trainingSize = self.Size
        testSize = X.shape[0]
        
        results = np.empty((testSize,))
        pbar = tqdm(
            total= testSize, initial= 0,
            unit_scale= True, leave= True)
        
        #X_train_PCA = np.int8(np.abs(PCA(self.X, 500)))
        #X_test_PCA = np.int8(np.abs(PCA(X, 500)))
        
        X_train_PCA = self.X.reshape(trainingSize,784)
        X_train_PCA = np.int8(judgeZero(X_train_PCA))
        X_test_PCA = X.reshape(testSize,784)
        X_test_PCA = np.int8(judgeZero(X_test_PCA)) 
        
        for index_test in range(0, testSize):
            distances = np.empty((trainingSize,))
            labelCounter = np.zeros((10,))
            
            diffMat = np.tile(X_test_PCA[index_test],(self.Size,1)) - X_train_PCA
            sqDiffMat = abs(diffMat)
            sqDistances = sqDiffMat.sum(axis= 1)
            distances = sqDistances
            distIndicies = np.argsort(distances)
            
            for i in range(self.k):
                label = self.y[distIndicies[i]]
                labelCounter[label] += 1
            predictLabel = labelCounter.argmax()
            results[index_test] = predictLabel
            pbar.update(1)
            
        return results
        # End of todo

if __name__ == '__main__':
    
    pass
