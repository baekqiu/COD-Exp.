import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def canny(img):
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #灰度化处理
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #高斯滤波降噪
    gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)

    #腐蚀化
    kernel = np.ones((5,5), np.uint8)
    k_ys = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)

    #Canny算子
    Canny = cv2.Canny(k_ys, 50, 150)

    return Canny

paths = glob.glob(r'/home/kai/Desktop/xxq/SINet-master/Dataset/TrainDataset2/Image/*.jpg')
paths.sort()

def picpro():
    num=0
    for files in paths:
        name=(files.split('/')[9]).split('.')[0]
        img = cv2.imread(files)
        opfile = r'/home/kai/Desktop/xxq/SINet-master/Dataset/TrainDataset2/Cannypic/'
        if (os.path.isdir(opfile) == False):
            os.mkdir(opfile)
        img = canny(img)
        img_path = opfile + str(name) + '.png'
        print(img_path)
        cv2.imwrite(img_path, img)
        num +=1
        print(num)
        print("批处理结束！")

if __name__ == '__main__':
    picpro()
           