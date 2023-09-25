import os
from torch.utils import data
import tqdm
from torchvision import transforms
from PIL import Image
import cv2

transform_default = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )

class AvatorSet(data.Dataset):
    def __init__(self, transform = transform_default, root = './faces/',
                 dataMode = 'single', slices = (0,70000)):
        super().__init__()
        self.dataMode = dataMode
        self.root = root
        self.transform = transform
        self.imageList = os.listdir(root)
        self.imageList = self.imageList[slices[0]:slices[1]]
        self.len_ = len(self.imageList)
        if dataMode == 'all':
            self.getData()
    
    def getData(self):
        pbar = tqdm.tqdm(self.imageList)
        self.dataSet = []
        pbar.set_description('Loading Data.')
        for imageName in pbar:
            fname = self.root + imageName
            image = cv2.imread(fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            self.dataSet.append(image)
            pass
        pass
    
    def __getitem__(self, index):
        #return super().__getitem__(index)
        if self.dataMode == 'single':
            fname = self.root + self.imageList[index]
            image = cv2.imread(fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            return image
        else: 
            return self.dataSet[index]
            
    
    def __len__(self):
        return self.len_
    pass

if __name__ == '__main__':
    #AvatorSet(slice= 20000)
    pass