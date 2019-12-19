import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
import cv2
import scipy as sp
import scipy.linalg as sl
import pylab
import random
import math

#Part 1
# X = np.arange(0, 5, 0.1)
# Z = [2 * x for x in X]
# Y = [np.random.normal(z, 0.5) for z in Z]
#
# plt.plot(X, Y, 'ro')
# plt.savefig('./2x.jpg')
# image = Image.open("./2x.jpg","r")
# line = (0, 0, 255, 1)
#
# for n in range(10):
#     i = random.randint(0,image.size[0]-10)
#     y = random.randint(0,image.size[1]-10)
#     for j in range(10):
#         for k in range(10):
#             image.putpixel((i + j, y + k), line)
# image.show()
# image.save("./2x.jpg")

#Part 处理图像
def img_handle(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)
    _, img_binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    edge = cv2.Canny(img_binary, 50, 150, apertureSize=3)
    img_binary = cv2.erode(img_binary, None, iterations=2)
    img_binary = cv2.dilate(img_binary, np.ones((5, 5), np.uint8), iterations=2)
    img_sobel = cv2.Sobel(img_binary,cv2.CV_64F,0,1,ksize=5)
    img_sobel = abs(img_sobel)
    _,numpy = cv2.threshold(img_sobel, 500, 255, cv2.THRESH_BINARY)
    return numpy,edge

#Part 2.1 最小二值化
def least_square_method(numpy):
    row, col = numpy.shape
    numpy = numpy[int(row / 5):int(row * 4 / 5), int(col / 3):int(2 * col / 3)]
    row, col = numpy.shape
    cv2.imshow('o', numpy)
    x = np.linspace(0, col, col)
    y = np.array(x)
    for i in range(col):
        numpy_y = row - np.argmax(numpy[:, i])
        y[i] = numpy_y
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    a0,a1 = np.linalg.solve(A,b)
    _X = [0, 5]
    _Y = [a0 + a1 * x for x in _X]
    plt.plot(_X, _Y, 'b', linewidth=2)
    plt.title("y = {} + {}x".format(a0, a1))
    plt.show()
    return x,y

#Part 2.2 RANSAC法
def RANSAC(SIZE,a,b,ite,sigma,p):
    X = np.linspace(0, 10, SIZE)
    Y = a * X + b
    fig = plt.figure()
    # 画图区域分成1行1列。选择第一块区域。
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("RANSAC")
    # 让散点图的数据更加随机并且添加一些噪声。
    random_x = []
    random_y = []
    for i in range(SIZE):
        random_x.append(X[i] + random.uniform(-0.5, 0.5))
        random_y.append(Y[i] + random.uniform(-0.5, 0.5))
    for i in range(SIZE):
        random_x.append(random.uniform(0, 10))
        random_y.append(random.uniform(10, 40))
    RANDOM_X = np.array(random_x)
    RANDOM_Y = np.array(random_y)
    # 画散点图。
    ax1.scatter(RANDOM_X, RANDOM_Y)
    # 横轴名称。
    ax1.set_xlabel("x")
    # 纵轴名称。
    ax1.set_ylabel("y")
    best_a1 = 0
    best_a0 = 0
    pretotal = 0
    for i in range(ite):
        sample_index = random.sample(range(SIZE * 2), 2)
        x_1 = RANDOM_X[sample_index[0]]
        x_2 = RANDOM_X[sample_index[1]]
        y_1 = RANDOM_Y[sample_index[0]]
        y_2 = RANDOM_Y[sample_index[1]]
        a1 = (y_2 - y_1) / (x_2 - x_1)
        a0 = y_1 - a1 * x_1
        total_inlier = 0
        for index in range(SIZE * 2):
            y_estimate = a1 * RANDOM_X[index] + a0
            if abs(y_estimate - RANDOM_Y[index]) < sigma:
                total_inlier = total_inlier + 1
        if total_inlier > pretotal:
            ite = math.log(1 - p) / math.log(1 - pow(total_inlier / (SIZE * 2), 2))
            pretotal = total_inlier
            best_a1 = a1
            best_a0 = a0
        if total_inlier > SIZE:
            break
    y = best_a1 * RANDOM_X + best_a0
    ax1.plot(RANDOM_X, y)
    text = "best_a1 = " + str(best_a1) + "\nbest_a0 = " + str(best_a0)
    plt.text(5, 10, text,fontdict={'size': 8, 'color': 'r'})
    plt.show()


#Part 2.3霍夫变换法
def hough_transform(img):
    img = ImageEnhance.Contrast(img).enhance(3)
    img = np.array(img)
    _,edges = img_handle(img)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=30, maxLineGap=18)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        pass
    img = Image.fromarray(img, 'RGB')
    img.show()

if __name__ == "__main__":
    # image = cv2.imread('./2x.jpg',1)
    # img = Image.open('./2x.jpg','r')
    image = cv2.imread('./simple.jpg', 1)
    img = Image.open('./simple.jpg', 'r')
    numpy,edge = img_handle(image)
    # least_square_method(numpy)
    # RANSAC(50,2,8,50000,0.25,0.99)
    hough_transform(img)
