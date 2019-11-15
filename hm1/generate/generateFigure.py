# !/usr/bin/zhb
# -*- coding = utf-8 -*-

from PIL import Image
import math

def generateFigure(imgW,imgH):
    img = Image.new("RGB",(imgW,imgH))
    background = (255,255,255,1)

    for i in range(imgW):
        for j in range(imgH):
            img.putpixel((i,j),background)

    red_line = (255,0,0,1)
    green_line = (0,255,0,1)
    blue_line = (0,0,255,1)

    x_width = (math.pi * 2) / imgW
    y_width =  imgH/4
    #正弦
    for i in range(imgW):
        y_value = int(math.sin(i*x_width)*y_width+y_width)
        img.putpixel((i,y_value),red_line)

    #余弦
    for i in range(imgW):
        y_value = int(math.cos(i*x_width)*y_width+y_width)
        img.putpixel((i,y_value),green_line)

    y_width2 = imgH/40
    #平方
    for i in range(imgW):
        y_value = -int(y_width2*math.pow(i*x_width,2))
        img.putpixel((i,y_value),blue_line)

    img.show()
    img.save("generateFigure.png")


generateFigure(1024,768)
