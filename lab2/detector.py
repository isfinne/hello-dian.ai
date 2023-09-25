import torch.nn as nn
import resnet
import torch
# TODO Design the detector.
# tips: Use pretrained `resnet` as backbone.

def axis2length(bbox):
    x1, y1, x1_, y1_ = bbox
    w = abs(x1_ - x1)
    h = abs(y1_ - y1)
    return (x1, y1, w, h)

"""DataFormat:
    X: a single batch of image: (nums, 3, 128, 128) -> resnet: (nums, 2048, 4, 4)
    y: bbox(x1,y1,x2,y2) + label(0,1,2,3,4) -> refers to five classes.     
"""

class Reshaper(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.bs = batch_size
    
    def __call__(self, X):
        return X.reshape(self.bs, -1)
    pass

class Detector(nn.Module):
    #loss_func = nn.SmoothL1Loss(reduction="sum")
    def __init__(self, lengths, num_classes, backbone='resnet18'):
        "lengths: 2048 * 4 * 4, 2048, 512"
        super().__init__()
        self.backbone = backbone
        self.lengths = lengths
        self.num_classes = num_classes
        if self.training != True:
            self.bs = 4
        else: self.bs = 32
        """
        self.resnetModel = resnet._resnet(arch= backbone, block= resnet.Bottleneck,
                                          layers= [3, 4, 6, 3], pretrained= False, progress= True,
                                          num_classes = 5)
        """
        netParam1 = 2048*1*1
        netParam2 = int(netParam1/2)
        bs = 16
        
        #self.resnetModel = resnet.resnet50(progress= False, num_classes = 5)
        self.resnetModel = resnet.resnet50(progress= False, num_classes = 5)
        #self.resnetModel = resnet.resnet18(progress= False, num_classes = 5)
        self.maxpool = nn.MaxPool2d(3,3)
        #self.reshaper32 = Reshaper(batch_size= 32)
        self.reshaper = Reshaper(batch_size= bs)
        self.reshaper4 = Reshaper(batch_size= 4)
        self.bn1 = nn.BatchNorm1d(netParam1)
        self.bn2 = nn.BatchNorm1d(netParam2)
        self.fc0 = nn.Linear(in_features= netParam1*1*1, out_features= netParam1)
        self.fc1 = nn.Linear(in_features= netParam1, out_features= netParam2)
        self.drp1 = nn.Dropout(p= 0.4)
        self.drp2 = nn.Dropout(p= 0.2)
        self.relu1 = nn.LeakyReLU()
        
        self.fc_cls_1 = nn.Linear(in_features= netParam2, out_features= netParam2 // 2 )
        self.fc_cls_2 = nn.Linear(in_features= netParam2 // 2, out_features= netParam2 // 2)
        self.fc_class = nn.Linear(in_features= netParam2 // 2, out_features= 5)
        
        self.fc_box_1 = nn.Linear(in_features= netParam2, out_features= netParam2 // 2 )
        self.fc_box_2 = nn.Linear(in_features= netParam2 // 2, out_features= netParam2 // 4)
        self.fc_bbox = nn.Linear(in_features= netParam2 // 4, out_features= 4)
        
        
        self.base = nn.Sequential(  self.resnetModel,
                                    self.maxpool,
                                    
                                    # not essential ?
                                    self.reshaper,
                                    self.fc0,
                                    self.drp2, 
                                    self.bn1,
                                    #self.drp1, # dropout at training 
                                    self.fc1, self.relu1, self.drp1, 
                                    # make sense ?
                                    self.bn2
                                    )
        self.cls = nn.Sequential(   self.fc_cls_1, self.relu1,
                                    self.drp2,
                                    self.fc_cls_2, self.relu1,
                                    self.drp2,
                                    self.fc_class)
        self.box = nn.Sequential(   self.fc_box_1, self.relu1,
                                    self.drp2,
                                    self.fc_box_2, self.relu1,
                                    self.fc_bbox)
        
        
    def forward(self, X):
        #batchNum = X.shape[0]
        """
        if self.training:
            Base = self.base_train(X)
        else: Base = self.base_border(X)
        """
        
        Base = self.base(X)
        probs_class = self.cls(Base)
        preds_box = self.box(Base)
        #probs_class = self.test2(X)
        #preds_box = self.test(X)
        #preds = self.base_all(X)
        return probs_class, preds_box #preds # preds[:,:5],  preds[:,5:]
    
    def visualize(self):
        
        pass
    
    pass


