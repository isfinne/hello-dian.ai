import numpy as np
#import torch as np
from tqdm import tqdm
import operator

class Knn(object):

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.Size = 60000

    def predict_internet(self, X):
        size = self.Size
        pbar = tqdm(
            total= size, initial= 0,
            unit= 'B', unit_scale= True, leave= True)
        
        dataSet = self.X
        labels = self.y
        k = self.k
        results = []
        for inX in X:
            dataSetSize = dataSet.shape[0]#查看矩阵的维度
            diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
            #tile(数组,(在行上重复次数,在列上重复次数))
            sqDiffMat = diffMat**2
            sqDistances = sqDiffMat.sum(axis=1)
            #sum默认axis=0，是普通的相加，axis=1是将一个矩阵的每一行向量相加
            distances = sqDistances**0.5
            sortedDistIndicies = distances.argsort()
            #sort函数按照数组值从小到大排序
            #argsort函数返回的是数组值从小到大的索引值
            classCount={}
            for i in range(k):
                voteIlabel = labels[sortedDistIndicies[i]]
                classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
                #get(key,k),当字典dic中不存在key时，返回默认值k;存在时返回key对应的值
            sortedClassCount = sorted(classCount.items(),
                key= operator.itemgetter(1),reverse=True)
            #python2中用iteritems，python3中用items代替；operator.itemgetter(k),返回第k个域的值
            results.append(sortedClassCount[0][0])
            pbar.update(1)
        return results

    
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
        
        results = np.empty((trainingSize,))
        pbar = tqdm(
            total= testSize, initial= 0,
            unit_scale= True, leave= True)
        #index_test = 0
        for index_test in range(0, testSize):
            distance = np.empty((trainingSize,))
            labelCounter = np.zeros((10,))
            for index_dist, Matrix_train in enumerate(self.X):
                #距离采用 p = 1 会快点吗? 
                distance[index_dist] = np.sum((Matrix_train - X[index_test]))
                pass
            distIndicies = np.argsort(distance)
            for i in range(self.k):
                label = self.y[distIndicies[i]]
                labelCounter[label] += 1
            predictLabel = labelCounter.argmax()
            results[index_test] = predictLabel
            pbar.update(1)
            
        return results
        #A single group of sum of distance: np.sum((X_train[0]-X_test[0])**0.5)
        
        pass
        # End of todo

if __name__ == '__main__':
    
    pass