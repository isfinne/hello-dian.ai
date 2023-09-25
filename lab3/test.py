from mmdet.apis import init_detector, inference_detector
#import mmcv

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = './tiny_vid/lizard/000003.JPEG'
result = inference_detector(model, img)

model.show_result(img, result)
model.show_result(img, result, out_file='./imageSave/result.jpg')