# 第二次作业报告

> 姓名 赵鸿博	学号 3017218180	

## 实验目的

* 为图像加上高斯噪声和椒盐噪声，再分别实现多种均值滤波器、统计排序滤波器和自适应滤波器对加了噪音后的图片进行还原处理，分析结果



## 实验过程

* 首先调用加噪音函数为图片分别加高斯噪声和椒盐噪声

```python
import numpy as np
import cv2
from numpy import shape
import random
from skimage.util import random_noise
from skimage import io
from tkinter import *

img = io.imread('1.JPG')
#高斯噪声
gauss_img = random_noise(img,mode='gaussian',seed=5000)
io.imsave('gauss_1.JPG',gauss_img)
#椒盐噪声
impulse_img = random_noise(img,mode='salt',seed=5000)
io.imsave('impulse_1.JPG',impulse_img)
```

* 处理图片，生成定义滤波器，分别对gauss加噪声和椒盐加噪声后图片进行处理

```python
def deal_image(path):
    image = io.imread(path, as_gray= True)
    med_img = io.imread(path, as_gray= True)	#中值滤波
    geometry_img = io.imread(path, as_gray= True)	#几何均值滤波
    mean_img = io.imread(path, as_gray= True)	#算数均值滤波
    max_img = io.imread(path, as_gray= True)	#最大值滤波
    min_img = io.imread(path, as_gray= True)	#最小值滤波
    mid_pot_img = io.imread(path, as_gray= True)	#中点滤波
    arf_img = io.imread(path, as_gray= True)	#修正后的阿尔法滤波
    xb_img = io.imread(path, as_gray= True)	#谐波滤波
    back_xb_img = io.imread(path, as_gray=True)	#反谐波滤波

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
image, med_img, mean_img, geometry_img, xb_img, back_xb_img, max_img, min_img, mid_pot_img, arf_img = deal_image('impulse_1.JPG')#impulse_1.JPG
```

###均值滤波器

* 均值滤波器都是使用3*3大小的滤波器用9个像素的均值代替中间的像素，用系数为1/mn(step)的卷积模板来实现  
* 实现算数均值滤波器

```python
#算数均值滤波器
def mean_filter(x, y, step):
    sum_s = 0
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += image[x + k][y + m] / (step * step)
    return sum_s
```

* 实现几何均值滤波器

```python
#几何均值滤波器
def geometry_filter(x, y, step):
    sum_s = 0
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s *= image[x + k][y + m]
    sum_r = sum_s ** (1/(step * step))
    return sum_r
```

* 实现谐波均值滤波器

```python
#谐波均值滤波器
def xb_filter(x, y, step):
    sum_s = 0
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += 1.0/image[x + k][y + m]
    sum_r = (step*step) / sum_s
    return sum_r
```

* 实现逆谐波均值滤波器

```python
#逆谐波均值滤波器
def back_xb_filter(x, y, step):
    sum_s = 0
    q = 1.5
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += image[x + k][y + m] / (step * step)
    sum_r = (sum_s ** (q+1)) / (sum_s ** q)
    return sum_r
```

### 统计排序滤波器

* 中值滤波器：最著名的顺序统计滤波器是中值滤波器，用该像素的相邻像素的灰度中值来替代该像素的值  

```python
#中值滤波器
def med_filter(x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(image[x + k][y + m])
    sum_s.sort()
    return sum_s[(int(step * step / 2) + 1)]
```

* 最大值滤波器

```python
#最大值滤波器
def max_filter(x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(image[x + k][y + m])
    sum_s.sort()
    return max(sum_s)
```

* 最小值滤波器

```python
#最小值滤波器
def min_filter(x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(image[x + k][y + m])
    sum_s.sort()
    return min(sum_s)
```

* 中点滤波器

```python
#中点滤波器
def mid_pot_filter(x, y, step):
    return 0.5 * (max_filter(x, y, step) + min_filter(x, y, step))
```

* 修正后的阿尔法均值滤波器

```python
#修正后的阿尔法均值滤波器
def arf_filter(x, y, step):
    sum_s = 0
    d = 5
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += image[x + k][y + m] / (step * step - d)
    return sum_s
```



* 设置滤波器大小，并调用各种滤波器处理图片，保存图片结果

```python
# Step为滤波器的大小 3*3
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
    
    #io.imsave(str(Step) + 'gauss_med.jpg', med_img)
    #io.imsave(str(Step) + 'gauss_mean.jpg', mean_img)
    #io.imsave(str(Step) + 'gauss_geometry.jpg', geometry_img)
    #io.imsave(str(Step) + 'gauss_xb.jpg', xb_img)
    #io.imsave(str(Step) + 'gauss_back_xb.jpg', back_xb_img)
    #io.imsave(str(Step) + 'gauss_max.jpg', max_img)
    #io.imsave(str(Step) + 'gauss_min.jpg', min_img)
    #io.imsave(str(Step) + 'gauss_midpoint.jpg', mid_pot_img)
    #io.imsave(str(Step) + 'gauss_arf.jpg', arf_img)

test(3)
```



## 实验结果对比

### 原图							Gauss加噪声               椒盐噪声

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/1.JPG" alt="1" style="zoom:33%;" /><img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/gauss_1.JPG" alt="gauss_1" style="zoom:33%;" /><img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/impulse_1.JPG" alt="impulse_1" style="zoom:33%;" />

### Gauss噪声还原

#### 算数均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_mean.jpg" alt="3gauss_mean" style="zoom: 50%;" />

#### 几何均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_geometry.jpg" alt="3gauss_geometry" style="zoom:50%;" />

#### 谐波均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_xb.jpg" alt="3gauss_xb" style="zoom:50%;" />

#### 逆谐波均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_back_xb.jpg" alt="3gauss_back_xb" style="zoom:50%;" />

#### 中值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_med.jpg" alt="3gauss_med" style="zoom:50%;" />

#### 最大值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_max.jpg" alt="3gauss_max" style="zoom:50%;" />

#### 最小值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_min.jpg" alt="3gauss_min" style="zoom:50%;" />

#### 中点滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_midpoint.jpg" alt="3gauss_midpoint" style="zoom:50%;" />

#### 修正后的阿尔法滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3gauss_arf.jpg" alt="3gauss_arf" style="zoom:50%;" />



### 椒盐噪声还原

#### 算数均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_mean.jpg" alt="3impulse_mean" style="zoom:50%;" />

#### 几何均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_geometry.jpg" alt="3impulse_geometry" style="zoom:50%;" />

#### 谐波均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_xb.jpg" alt="3impulse_xb" style="zoom:50%;" />

#### 逆谐波均值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_back_xb.jpg" alt="3impulse_back_xb" style="zoom:50%;" />

#### 中值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_med.jpg" alt="3impulse_med" style="zoom:50%;" />

#### 最大值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_max.jpg" alt="3impulse_max" style="zoom:50%;" />

#### 最小值滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_min.jpg" alt="3impulse_min" style="zoom:50%;" />

#### 中点滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_midpoint.jpg" alt="3impulse_midpoint" style="zoom:50%;" />

#### 修正后的阿尔法滤波处理

<img src="/Users/zhb/Desktop/大三课程/图像处理/作业2/3impulse_arf.jpg" alt="3impulse_arf" style="zoom:50%;" />

