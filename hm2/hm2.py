import numpy as np
import cv2
from numpy import shape
import random
from skimage.util import random_noise
from skimage import io
from tkinter import *

img = io.imread('1.JPG')

# #高斯噪声
# gauss_img = random_noise(img,mode='gaussian',seed=5000)
# io.imsave('gauss_1.JPG',gauss_img)
# #椒盐噪声
# impulse_img = random_noise(img,mode='salt',seed=5000)
# io.imsave('impulse_1.JPG',impulse_img)


def deal_image(path):
    image = io.imread(path, as_gray= True)
    med_img = io.imread(path, as_gray= True)
    geometry_img = io.imread(path, as_gray= True)
    mean_img = io.imread(path, as_gray= True)
    max_img = io.imread(path, as_gray= True)
    min_img = io.imread(path, as_gray= True)
    mid_pot_img = io.imread(path, as_gray= True)
    arf_img = io.imread(path, as_gray= True)
    xb_img = io.imread(path, as_gray= True)
    back_xb_img = io.imread(path, as_gray=True)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            med_img[i][j] = image[i][j]
            geometry_img[i][j] = image[i][j]
            mean_img[i][j] = image[i][j]
            xb_img[i][j] = image[i][j]
            back_xb_img[i][j] = image[i][j]
            max_img[i][j] = image[i][j]
            min_img[i][j] = image[i][j]
            mid_pot_img[i][j] = image[i][j]
            arf_img[i][j] = image[i][j]
    return image, med_img, mean_img, geometry_img, xb_img, back_xb_img, max_img, min_img, mid_pot_img, arf_img
#定义滤波器
image, med_img, mean_img, geometry_img, xb_img, back_xb_img, max_img, min_img, mid_pot_img, arf_img = deal_image('impulse_1.JPG')

#算数均值滤波器
def mean_filter(x, y, step):
    sum_s = 0
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += image[x + k][y + m] / (step * step)
    return sum_s

#几何均值滤波器
def geometry_filter(x, y, step):
    sum_s = 0
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s *= image[x + k][y + m]
    sum_r = sum_s ** (1/(step * step))
    return sum_r

#谐波均值滤波器
def xb_filter(x, y, step):
    sum_s = 0
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += 1.0/image[x + k][y + m]
    sum_r = (step*step) / sum_s
    return sum_r

#逆谐波均值滤波器
def back_xb_filter(x, y, step):
    sum_s = 0
    q = 1.5
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += image[x + k][y + m] / (step * step)
    sum_r = (sum_s ** (q+1)) / (sum_s ** q)
    return sum_r

#中值滤波器
def med_filter(x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(image[x + k][y + m])
    sum_s.sort()
    return sum_s[(int(step * step / 2) + 1)]



#最大值滤波器
def max_filter(x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(image[x + k][y + m])
    sum_s.sort()
    return max(sum_s)

#最小值滤波器
def min_filter(x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(image[x + k][y + m])
    sum_s.sort()
    return min(sum_s)

#中点滤波器
def mid_pot_filter(x, y, step):
    return 0.5 * (max_filter(x, y, step) + min_filter(x, y, step))

#修正后的阿尔法均值滤波器
def arf_filter(x, y, step):
    sum_s = 0
    d = 5
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += image[x + k][y + m] / (step * step - d)
    return sum_s

# Step为滤波器的大小
def test(Step):
    for i in range(int(Step / 2), image.shape[0] - int(Step / 2)):
        for j in range(int(Step / 2), image.shape[1] - int(Step / 2)):
            med_img[i][j] = med_filter(i, j, Step)
            mean_img[i][j] = mean_filter(i, j, Step)
            geometry_img[i][j] = mean_filter(i,j,Step)
            xb_img[i][j] = xb_filter(i, j, Step)
            back_xb_img[i][j] = back_xb_filter(i, j, Step)
            max_img[i][j] = max_filter(i, j, Step)
            min_img[i][j] = min_filter(i, j, Step)
            mid_pot_img[i][j] = mid_pot_filter(i,j,Step)
            arf_img[i][j] = arf_filter(i, j, Step)
    io.imsave(str(Step) + 'impulse_med.jpg', med_img)
    io.imsave(str(Step) + 'impulse_mean.jpg', mean_img)
    io.imsave(str(Step) + 'impulse_geometry.jpg', geometry_img)
    io.imsave(str(Step) + 'impulse_xb.jpg', xb_img)
    io.imsave(str(Step) + 'impulse_back_xb.jpg', back_xb_img)
    io.imsave(str(Step) + 'impulse_max.jpg', max_img)
    io.imsave(str(Step) + 'impulse_min.jpg', min_img)
    io.imsave(str(Step) + 'impulse_midpoint.jpg', mid_pot_img)
    io.imsave(str(Step) + 'impulse_arf.jpg', arf_img)

test(3)


