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
from detect_single_img import detect
from camera_stick_simulation import clip_POP, camera_sticker_intensity_adjustment, img_camera_sticker_pattern, initiation_POP_ablation, get_p_best, get_G_best, initiation_V, update_V, update_POP

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

image_path= r'C:\Users\a\Desktop\TT100kcoco\mnt\data0\home\zhoujingzhi\detection_dataset\TT100K-coco\images\test'#图片的路径
label_path='TT100K-1/labels'#标签的路径
image_dirs=os.listdir(image_path)
label_dirs=os.listdir(label_path)
image_dirs.sort()
label_dirs.sort()

Step = 5
POP_size = 100
N = 5

Omega = 0.9
c1, c2 = 1.6, 2.0


asr = np.zeros((1, 27))
query = np.zeros((1, 27))

for Color in range(0, 27):
    ALPHA = 0.5
    WIDTH = 0.1

    count_all = 0
    for img_id in range(0, 970):

        # print(image_dirs[img_id])
        # print(label_dirs[img_id])

        singleimage = image_path + "/" + image_dirs[img_id]
        # singlelabel = label_path + "/" + label_dirs[img_id]

        # with open(singlelabel, "r") as f:
        #     label = f.readline()
        # f.close()
        # label = label[:-1].split(" ")[0]
        # print('label = ', label)

        label_clean = detect(singleimage)
        (a, b) = label_clean.shape
        print('img_id, count_all, label_clean.shape = ', img_id, count_all, label_clean.shape)

        if a != 1:
            continue

        count_all = count_all + 1

        POP = np.zeros((POP_size, N + 1))  # 最后一位存储置信度
        POP_best = np.zeros((POP_size, N + 1))  # 最后一位存储置信度
        G_best = np.ones((1, N + 1))
        V = np.zeros((POP_size, N))
        # print('G_best = ', G_best)

        POP = initiation_POP_ablation(Color, POP_size, N)
        # print(POP)
        for i in range(0, POP_size):
            for j in range(N + 1):
                POP_best[i][j] = POP[i][j]

        print(POP_best)

        V = initiation_V(V)

        print('V = ', V)

        for step in range(Step):

            tag_break = 0

            print('asr = ', asr)
            print('query = ', query)

            for population in range(POP_size):

                query[0][Color] = query[0][Color] + 1

                image = cv2.imread(singleimage)
                print('Color, img_id, count_all, step, population = ', Color, img_id, count_all, step,
                      population)
                path_adv = 'adv.jpg'
                img_camera_sticker_pattern(image, POP[population][3], 0, POP[population][4], 2048,
                                           POP[population][0], POP[population][1], POP[population][2],
                                           int(WIDTH * 2048), path_adv)

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
                        frame = camera_sticker_intensity_adjustment(frame, i % 5, ALPHA, path_adv)
                        # cv2.imshow('video', frame)
                        videoWriter.write(frame)
                        i += 1
                        c = cv2.waitKey(1)
                        if c == 27:
                            break
                    else:
                        break

                # img_show = plt.imread(adv)
                # plt.imshow(img_show)
                # plt.show()

                label_adv = detect(path_adv)
                # print('label_adv.shape = ', label_adv.shape)
                if label_adv.shape == (a - 1, 6) or label_adv.shape == (0, 6):  # 判断是否攻击成功
                    tag_break = 1
                    asr[0][Color] = asr[0][Color] + 1
                    break

                # print('label_adv[0][4] = ', label_adv[0][4])

                # print('label_adv = ', label_adv)
                # print('label_adv.shape = ', label_adv.shape)
                # print('population, N = ', population, N)
                POP[population][N] = label_adv[0][4]

            if tag_break == 1:
                break

            POP_best = get_p_best(POP, POP_best)
            G_best = get_G_best(G_best, POP_best)
            # print('POP = ', POP)
            # print('POP_best = ', POP_best)
            # print('G_best = ', G_best)
            # print('V = ', V)
            V = update_V(V, Omega, c1, c2, POP, POP_best, G_best)
            # print('V = ', V)
            # print('POP = ', POP)
            POP = update_POP(POP, V)
            # print('POP = ', POP)
            POP = clip_POP(POP)
            # print('POP = ', POP)

        # if count_all == 3:
        #     break

print('asr = ', asr)
print('query = ', query)

print('asr/2.72 = ', asr/2.72)
print('query/272 = ', query/272)