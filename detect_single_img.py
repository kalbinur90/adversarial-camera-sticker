import cv2
import random
import numpy as np
import os
import torch
from detect import run
import argparse
import platform
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from urllib.request import urlopen

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
attack_times = []
error_rate = 0
originPath=os.getcwd()
image_path='TT100K-1/images'#图片的路径
label_path='TT100K-1/labels'#标签的路径
image_dirs=os.listdir(image_path)
label_dirs=os.listdir(label_path)
image_dirs.sort()
label_dirs.sort()
ALPHA = 0.1 # 0.1-0.9
WIDTH = 0.2 # 0.1-0.7
result_path = 'sticker_results/exp_' + str(int(ALPHA * 10)) + '_' + str(int(WIDTH * 10)) + '.txt'
#camera sticker simulation


def img_camera_sticker_pattern(img, x1, y1, x2, y2, b, g, r, thickness, path):


    cv2.line(img, (x1, y1), (x2, y2), (r, g, b), thickness)

    cv2.imwrite(path, img)

#camera sticker intensity adjustment
def camera_sticker_intensity_adjustment(img, cnt, I, path_adv):

    if cnt == 0:
        return img

    height, width, n = img.shape

    mask = {
        1: cv2.imread(path_adv),
    }

    mask[cnt] = cv2.resize(mask[cnt], (width, height), interpolation=cv2.INTER_CUBIC)

    new_img = cv2.addWeighted(img, (1 - I), mask[cnt], I, 0)

    return new_img

def parse_opt(image):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / r'C:\Users\a\PycharmProjects\pythonProject\KALI1\yolov5\best1.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=image, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/rett100k.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default='True',action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def detect(path):
    opt = parse_opt(path)
    pred = run(**vars(opt))
    pred1 = pred[0].cpu().numpy()
    # print('pred1', pred1)
    return pred1
#
# res = detect('1.jpg')
# print(res)


# def detect(path):
#     model = torch.hub.load("ultralytics/yolov5", "custom", path = r"C:\Users\a\PycharmProjects\pythonProject\yolov5\best1.pt", device = '0')
#     result = model(path)
#     result.save()
#     return result
#
# re = detect(r'C:\Users\a\Desktop\TT100kcoco\mnt\data0\home\zhoujingzhi\detection_dataset\TT100K-coco\images\test\95153.jpg')
# re1 = detect(r'C:\Users\a\PycharmProjects\pythonProject\yolov5\adv\transfer_attack\95153.jpg')



# import torch
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
# img = '1.jpg'
# result = model(img)
#
# result.print()
#
# print(result.xyxy[0])



















