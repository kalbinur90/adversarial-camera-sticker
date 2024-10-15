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
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best1.pt', help='model path or triton URL')
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

def run_ablation(epoch, image, label, var_width, var_alpha, opt):
    
    for step in range(0, 100):

        w, h = image.shape[0], image.shape[1]
        # print(w, h)

        #半透明相机贴片的物理参数
        x1, y1, x2, y2 = random.randint(0, w), 0, random.randint(0, w), h #起点与终点
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)#颜色
        I = var_alpha #透明度
        thickness = int(var_width * w)#宽度

        img = cv2.imread(singleimage)

        path_adv ='sticker_results/temp_' + str(int(ALPHA * 10)) + '_' + str(int(WIDTH * 10)) +'.jpg'
        img_camera_sticker_pattern(img, x1, y1, x2, y2, b, g, r, thickness, path_adv)
        

        #生成半透明效果
        
        cap = cv2.VideoCapture(singleimage)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        i = 1
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(path_adv, fourcc, fps, (width, height))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = camera_sticker_intensity_adjustment(frame, i % 5, I, path_adv)
                # cv2.imshow('video', frame)
                videoWriter.write(frame)
                i += 1
                c = cv2.waitKey(1)
                if c == 27:
                    break
            else:
                break



        print('epoch, step', epoch, step)
        # img_show = plt.imread(adv)
        # plt.imshow(img_show)
        # plt.show()

        opt = parse_opt(path_adv)
        pred = run(**vars(opt))
        pred = pred[0].cpu().numpy()
        os.remove(path_adv)
        success = False
        if len(pred) == 0:
            success = True
        elif int(pred[0][-1]) != int(label[0]):
            success = True
        
        if success:
            state = [epoch, step + 1, x1, y1, x2, y2, r, g, b, var_width, var_alpha]
            return True, state
    
    return False, []


for epoch in range(len(image_dirs)):
    
    print(epoch + 1,'/',len(image_dirs))
    singleimage = originPath + "/" + image_path + "/" + image_dirs[epoch]
    singlelabel = originPath + "/" + label_path + "/" + label_dirs[epoch]
    #singleimage = originPath + "/" + image_path + "/" + '10009.jpg'
    #singlelabel = originPath + "/" + label_path + "/" + '10009.txt'
    print(singleimage,singlelabel)
    image = cv2.imread(singleimage)
    with open(singlelabel, "r") as f:
        label = f.readline()
    f.close()
    label = label[:-1].split(" ")

    # 判定
    # /mnt/data0/home/zhoujingzhi/yolov5/
    #model = torch.hub.load('/mnt/data0/home/zhoujingzhi/yolov5/', 'custom','/mnt/data0/home/zhoujingzhi/yolov5/best.pt', source='local')
    opt = parse_opt(singleimage)
    
    pred = run(**vars(opt))

    pred = pred[0].cpu().numpy()
    if len(pred) == 0:
        error_rate += 1
        continue
    else:
         pred=pred[0]
         if int(pred[-1]) != int(label[0]):
           error_rate += 1
           print(int(pred[-1]),label[0])
           continue

    flag, state = run_ablation(epoch, image, label, WIDTH, ALPHA, opt)

    if flag is True:
        attack_times.append(state[1])
        f = open(result_path, 'a')
        f.write(str(state) + '\n')
        f.close()
        


print(error_rate)
print(len(attack_times))
print(sum(attack_times) / len(attack_times))
cso = [error_rate, len(attack_times), sum(attack_times) / len(attack_times)]
f = open(result_path, 'a')
f.write(str(cso) + '\n' + str(attack_times) + '\n\n')
f.close()





