import numpy as np
import torch
def iou_single(x1,y1, x2, y2, a1, b1, a2, b2):
    ax = max(x1, a1) # 相交区域左上角横坐标
    ay = max(y1, b1) # 相交区域左上角纵坐标
    bx = min(x2, a2) # 相交区域右下角横坐标
    by = min(y2, b2) # 相交区域右下角纵坐标

    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)

    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h

    return area_X / (area_N + area_M - area_X)


def compute_iou(bbox1, bbox2):
    # TODO Compute IoU of 2 bboxes.
    x1, y1, x1_, y1_ = bbox1[:,0:1], bbox1[:,1:2], bbox1[:,2:3], bbox1[:,3:4]
    x2, y2, x2_, y2_ = bbox2[:,0:1], bbox2[:,1:2], bbox2[:,2:3], bbox2[:,3:4]
    x12 = torch.max(x1, x2)
    y12 = torch.max(y1, y2)
    x12_ = torch.min(x1_, x2_)
    y12_ = torch.min(y1_, y2_)
    width = torch.max(torch.zeros_like(x12), x12_ - x12)
    height = torch.max(torch.zeros_like(x12), y12_ - y12)
    Intersection = width * height
    Union = torch.abs((x1_-x1) * (y1_-y1)) + torch.abs((x2_-x2) * (y2_-y2)) - Intersection
    IoU = Intersection / Union
    return IoU
    # End of todo

def compute_iou_torch(bbox1, bbox2):
    import torchvision 
    return torchvision.ops.box_iou(bbox1, bbox2)

def compute_iou_refer(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(  
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  
    wh[wh < 0] = 0 
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    result = torch.diagonal(iou)
    return result

if __name__ == '__main__':
    # debug
    a = iou_single(0,0,115,126,20,4,119,123)
    pass 