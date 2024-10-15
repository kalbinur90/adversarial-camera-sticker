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


def color(N):
    r, g, b = 0, 0, 0
    if N == 0:
        r, g, b = 0, 0, 0
    if N == 1:
        r, g, b = 0, 0, 127
    if N == 2:
        r, g, b = 0, 0, 255
    if N == 3:
        r, g, b = 0, 127, 0
    if N == 4:
        r, g, b = 0, 127, 127
    if N == 5:
        r, g, b = 0, 127, 255
    if N == 6:
        r, g, b = 0, 255, 0
    if N == 7:
        r, g, b = 0, 255, 127
    if N == 8:
        r, g, b = 0, 255, 255
    if N == 9:
        r, g, b = 127, 0, 0
    if N == 10:
        r, g, b = 127, 0, 127
    if N == 11:
        r, g, b = 127, 0, 255
    if N == 12:
        r, g, b = 127, 127, 0
    if N == 13:
        r, g, b = 127, 127, 127
    if N == 14:
        r, g, b = 127, 127, 255
    if N == 15:
        r, g, b = 127, 255, 0
    if N == 16:
        r, g, b = 127, 255, 127
    if N == 17:
        r, g, b = 127, 255, 255
    if N == 18:
        r, g, b = 255, 0, 0
    if N == 19:
        r, g, b = 255, 0, 127
    if N == 20:
        r, g, b = 255, 0, 255
    if N == 21:
        r, g, b = 255, 127, 0
    if N == 22:
        r, g, b = 255, 127, 127
    if N == 23:
        r, g, b = 255, 127, 255
    if N == 24:
        r, g, b = 255, 255, 0
    if N == 25:
        r, g, b = 255, 255, 127
    if N == 26:
        r, g, b = 255, 255, 255

    return r, g, b


def img_camera_sticker_pattern(img, x1, y1, x2, y2, b, g, r, thickness, path):


    x1, y1, x2, y2, r, g, b = int(x1), int(y1), int(x2), int(y2), int(r), int(g), int(b)
    # print('r, g, b = ', r, g, b)
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

def initiation_POP(POP_size, N):
    POP = np.zeros((POP_size, N + 1))
    for i in range(0, POP_size):
        POP[i][0] = random.randint(0, 255)
        POP[i][1] = random.randint(0, 255)
        POP[i][2] = random.randint(0, 255)
        POP[i][3] = random.randint(0, 2048)
        POP[i][4] = random.randint(0, 2048)
        POP[i][5] = 100
    return POP

def initiation_POP_ablation(Col, POP_size, N):
    POP = np.zeros((POP_size, N + 1))
    r, g, b = color(Col)
    for i in range(0, POP_size):
        POP[i][0] = r
        POP[i][1] = g
        POP[i][2] = b
        POP[i][3] = random.randint(0, 2048)
        POP[i][4] = random.randint(0, 2048)
        POP[i][5] = 100
    return POP

def initiation_V(V):
    a, b = V.shape
    for i in range(0, a):
        V[i][0] = random.randint(-127, 127)
        V[i][1] = random.randint(-127, 127)
        V[i][2] = random.randint(-127, 127)
        V[i][3] = random.randint(-1024, 1024)
        V[i][4] = random.randint(-1024, 1024)

    return V


def get_p_best(POP, POP_best):
    a, b = POP.shape
    for i in range(a):
        if POP[i][b-1] < POP_best[i][b-1]:
            for j in range(b):
                POP_best[i][j] = POP[i][j]

    return POP_best

def get_G_best(G_Best, POP_best):
    a, b = POP_best.shape
    for i in range(a):
        if G_Best[0][b-1] > POP_best[i][b-1]:
            for j in range(b):
                G_Best[0][j] = POP_best[i][j]

    return G_Best

def update_V(V, Omega, c1, c2, POP, POP_best, G_best):

    a, b = POP.shape
    tag_POP, tag_POP_best, tag_G_best = np.zeros((a, b-1)), np.zeros((a, b-1)), np.zeros((a, b-1))

    for i in range(a):
        for j in range(b-1):
            tag_POP[i][j] = POP[i][j]
            tag_POP_best[i][j] = POP_best[i][j]
            tag_G_best[i][j] = G_best[0][j]

    # print('tag_POP = ', tag_POP)
    # print('tag_POP_best = ', tag_POP_best)
    # print('tag_G_best = ', tag_G_best)

    V = Omega * V + c1*(tag_POP_best - tag_POP) + c2*(tag_G_best - tag_POP)
    for i in range(a):
        for j in range(b-1):
            V[i][j] = int(V[i][j])

    return V

def update_POP(POP, V):
    a, b = V.shape
    for i in range(a):
        for j in range(b):
            POP[i][j] = POP[i][j] + V[i][j]

    return POP

def clip_POP(POP):
    a, b = POP.shape
    for i in range(a):
        for j in range(0, 3):
            if POP[i][j] > 255 or POP[i][j] < 0:
                POP[i][j] = random.randint(0, 255)
        for j in range(3, 5):
            if POP[i][j] > 2048 or POP[i][j] < 0:
                POP[i][j] = random.randint(0, 2048)
    return POP























