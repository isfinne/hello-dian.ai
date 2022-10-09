import os
from torch.utils import data
import transforms
import torch


class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.
    
    def __init__(self, root = './tiny_vid/', mode = 'train'):
        import re
        #super().__init__()
        self.root = root
        dirs = os.listdir(root)
        self.labelPaths = []
        self.labels = {}
        self.images = {}
        for file in dirs:
            try:
                path = re.search('(\w+)_gt.txt', file)
                self.labelPaths.append(path.group(0))
                class_ = path.group(1)
                with open(root + file, 'r') as f:
                    labelGroup = f.read().split('\n')
                    self.labels[class_] = labelGroup[:180]
            except:
                if os.path.isdir(root + file):
                    imageList = os.listdir(root + file)
                    self.images[file] = imageList[:180]
        self.Range_ = (0,160) if mode == 'train' else (148, 180)
        self.train = True if mode == 'train' else False
        self.getDataSet()
    
    def getDataSet(self):
        toTensor = transforms.ToTensor()
        loader = transforms.LoadImage()
        filper = transforms.RandomHorizontalFlip()
        croper = transforms.RandomCrop((128, 128))
        keys = self.labels.keys()
        _classDict = {k :v for k,v in zip(keys, range(5))}
        self.dataSet = []
        for class_ in keys:
            imageNameList = self.images[class_]
            labelsList = self.labels[class_]
            for index in range( *self.Range_ ):
                imagePath = self.root + class_ + '/' + imageNameList[index]
                label = list(map(lambda x: int(x), labelsList[index].split(' ')[1:]))
                #outList = [ 1 if i==_classDict[class_] else 0 for i in range(5)]
                image, bbox = loader(imagePath, label)
                if self.train:
                    image, bbox = filper(image, bbox)
                    image, bbox = croper(image, bbox)
                image, bbox = toTensor(image, bbox)
                self.dataSet.append((image, bbox, torch.tensor(_classDict[class_])))
                pass
            pass
        return self.dataSet
        
    def __getitem__(self, index):
        #return super().__getitem__(index)
        if index >= len(self):
            raise IndexError
        return self.dataSet[index]
    
    def __len__(self):
        return len(self.dataSet)

    # End of todo

"""DataSet:
    800 Groups
    for each group (len = 3):
    image: 3 x 128 x 128
    bbox: 4
    label: 1 (int, range: 0-5)
"""
if __name__ == '__main__':
    dataset = TvidDataset(root='./tiny_vid/', mode='train')
    #print(len(dataset[0]))
    #print(len(dataset.dataSet))
    import pdb; pdb.set_trace()
