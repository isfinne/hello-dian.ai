"""_tempRecord_
    GPU environment:        pytorch_gpu (failed)
    use GPU Command:        conda activate pytorch_gpu
                            python ./main.py 
    acc_strict definition:  IOU >= 0.5 and the classification is correct   
    
"""
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from tvid import TvidDataset
from detector import Detector
from utils import compute_iou

#lr = 5e-3
lr = 6e-3
#batch = 32
batch = 16
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)
iou_thr = 0.5

def train_epoch(model, dataloader, criterion: dict, optimizer,
                scheduler, epoch, device):
    model.to(device)
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    for X, y, class_ in bar:
        X = X.to(device)
        y = y.to(device)
        class_ = class_.to(device)
        # TODO Implement the train pipeline.
        # class_pred, bbox = model.forward(X)
        if X.shape[0] != batch:
            break
        class_probs, bbox_pred = model.forward(X)
        #loss = criterion['cls'](class_probs, class_.long()) + 0.2*criterion['box'](bbox_pred, y) 
        #loss = 1.2*criterion['cls'](class_probs, class_.long()) + 0.25*criterion['box'](bbox_pred, y) 
        
        loss = 0.3*criterion['box'](bbox_pred, y) + 2.7*criterion['cls'](class_probs, class_)
        #loss = 2.7*criterion['cls'](class_probs, class_)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print(torch.argmax(class_pred, dim= 1))
        #print(class_)
        class_preds = torch.argmax(class_probs, dim= 1)
        class_correct = torch.sum(class_ == class_preds) 
        
        total = len(class_)
        IOU = compute_iou(bbox_pred, y)
        #IOU1 = compute_iou(bbox_pred, y)
        class_delta = class_preds - class_
        correct = sum( [1 if delta == 0 and iou >= 0.5 else 0 for delta, iou in zip(class_delta, IOU)] )
        avgIOU = torch.sum(IOU)/batch
        bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.4f}'
                            f' clsacc={class_correct // total * 100:.1f}'
                            f' stracc = {correct // total *100:.1f}'
                            f' loss={loss.item():.1f}'
                            f' IOU={avgIOU:.2f}'
                            )
        # End of todo
    scheduler.step()


def test_epoch(model, dataloader, device, epoch):
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct, correct_cls, total = 0, 0, 0
        correct_IOU = 0
        totalIOU = 0
        for X, y, class_ in dataloader:
            X = X.to(device)
            y = y.to(device)
            class_ = class_.to(device)
            # TODO Implement the test pipeline.
            class_probs, bbox_pred = model.forward(X)
            class_preds = torch.argmax(class_probs, dim= 1)
            class_delta = class_preds - class_
            
            IOU = compute_iou(bbox_pred, y)
            totalIOU_1 = torch.sum(IOU)
            totalIOU += totalIOU_1
            
            correct_single = sum([1 if delta == 0 and iou >= 0.45 else 0 for delta, iou in zip(class_delta, IOU)])
            correct += correct_single
            
            correct_IOU_single = sum([1 if iou >= 0.45 else 0 for iou in IOU])
            correct_IOU += correct_IOU_single
            
            correct_cls_single = sum( [1 if delta == 0 else 0 for delta in class_delta ])
            correct_cls += correct_cls_single
            
            total += X.shape[0]
            # End of todo

        print(
              f' val clsacc: {correct_cls / total * 100:.2f} '
              f' val IOUacc: {correct_IOU / total * 100:.2f}'
              f' val AvgIOU :{totalIOU / total:.2f}')


def main():
    trainloader = data.DataLoader(TvidDataset(root='./tiny_vid/', mode='train'),
                                  batch_size=batch, shuffle=True, num_workers=4)
    #trainloader = DataLoader(load_mnist(), batch= batch)
    testloader = data.DataLoader(TvidDataset(root='./tiny_vid/', mode='test'),
                                 batch_size=batch, shuffle=True, num_workers=4)
    model = Detector(backbone='resnet50', lengths=(2048 * 4 * 4, 2048, 512),
                     num_classes=5).to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95,
                                                last_epoch=-1)
    criterion = {'cls': nn.CrossEntropyLoss(), 'box': nn.L1Loss()}

    for epoch in range(epochs):
        train_epoch(model, trainloader, criterion, optimizer,
                    scheduler, epoch, device)
        test_epoch(model, testloader, device, epoch)


if __name__ == '__main__':
    main()
