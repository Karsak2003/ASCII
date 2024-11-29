
#region IMPORT
from math import ceil
import sys
from typing import Any, Iterable, Sequence
import cv2
import numpy as np  
import os
from PIL import Image
from progress.bar import IncrementalBar, ChargingBar
import time

import matplotlib as mpl
import matplotlib.pyplot as plt  
from matplotlib.figure import Figure 
from matplotlib.axes import Axes 
import matplotlib.animation as animation

# from PySide6 import QtWidgets, QtCore, QtGui

from tkinter import filedialog as tk_filedialog
#endregion IMPORT


#__________________________________________________________________________________
asii = """$@B%8&WM*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`'."""
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_1 = """`.-':,^=;><+!rc*z?sLTvJ7FiCfI31tluneoZ5Yxjya2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"""[::-1]
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_2 = """$@B%8&WM#oahkbdpqwmZO0QLCJUYXzcvunxrjf1?+~ilI;:*^"',."""
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_3 = r" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
asii_3 = r" .',:`;\"i!I^lr1vjcx<>Yft*JL?T7uynozaksFVXeh3Cq2KUdp4SZbA0w5GPg9EOH6mDQNR8%&BWM#@$"
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

CWD = os.getcwd()
FRESU = CWD + "\\~resu\\"
FTEMP = CWD + "\\~temp\\"

#region Matrix
kernel2 = np.ones((5, 5), np.float32)/25
Gx:np.ndarray               = np.array([[1,  0, -1],[2,  0, -2],[1,  0, -1]])
Gy:np.ndarray               = np.array([[1,  2, 1],[0,  0, 0],[-1,  -2, -1]])
MeanMat:np.ndarray          = [1/3, 1/3, 1/3]
RedMat:np.ndarray           = [[1, 0, 0],[0, 0, 0],[0, 0, 0]]
GreenMat:np.ndarray         = [[0, 1, 0],[0, 0, 0],[0, 0, 0]]
BlueMat:np.ndarray          = [[0, 0, 1],[0, 0, 0],[0, 0, 0]]
MeanMat2Red:np.ndarray      = [[1/3, 0, 0],[1/3, 0, 0],[1/3, 0, 0]]
MeanMat2Green:np.ndarray    = [[0, 1/3, 0],[0, 1/3, 0],[0, 1/3, 0]]
MeanMat2Blue:np.ndarray     = [[0, 0, 1/3],[0, 0, 1/3],[0, 0, 1/3]]
MeanMat2RGB:np.ndarray      = [[1/3, 1/3, 1/3],[1/3, 1/3, 1/3],[1/3, 1/3, 1/3]]
#endregion Matrix

#region IMG2
img2gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
img2bgr  = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
img2rgb  = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
img2CRev = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
img2not  = lambda x: cv2.bitwise_not(x)
img2and  = lambda x, y: cv2.bitwise_and(x, y) 
img2or   = lambda x, y: cv2.bitwise_or(x, y)
img2xor  = lambda x, y: cv2.bitwise_xor(x, y)
img2blur = lambda x, a=7, b=7, s=0: cv2.GaussianBlur(x, (a, b), s)
img2cont = lambda x: cv2.Canny(x, 100, 160)
#// img2Sobel = lambda x, dx = 1, dy = 1: cv2.Sobel(x, ddepth=cv2.CV_64F, dx=dx, dy=dy, ksize=3, scale=0.1)
#// img2SobelComb = lambda x: cv2.addWeighted(img2Sobel(x, dy=0), 0.5, img2Sobel(x, dx=0), 0.5, 0)
#// img2Apletude = lambda x: np.sqrt(img2Sobel(x, dy=0)**2 + img2Sobel(x, dx=0)**2)
#// img2Angel = lambda x: np.arctan2(img2Sobel(x, dy=0), img2Sobel(x, dx=0))/np.pi * 0.5 + 0.5
img2DoG = lambda x, a=((3, 3),(7, 7)), s=0., t=1.0: (1+t)*img2blur(x, a[0][0], a[0][1], s)-t*img2blur(x, a[1][0], a[1][1], s)
img2Laplacian = lambda x: cv2.Laplacian(x, cv2.CV_64F)

Img2quantize = lambda x, k: np.ceil(x/k + 0.5)*k
RGB_Img2quantize = lambda x, k: np.ceil(x/k + 0.5)*k
img2filter2D = lambda x, a=kernel2: cv2.filter2D(src=x, ddepth=-1, kernel=a)
#endregion IMG2

Sigmoid = lambda x: 1/(1 + np.e**(-x))



def ImgShow(image) -> None:
    f, a = plt.subplots()
    axs:Axes = a
    fig:Figure = f
    del a, f
    axs.imshow(image)
    plt.show()

def _txt():
    temp = dict()
    
    for c in asii_3:
        outImg:np.ndarray = np.zeros((27, 25, 3), np.uint8)
        cv2.putText(outImg, c, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255))
        temp[c] = outImg.sum()
        #ImgShow(outImg)
        cv2.imwrite(FTEMP+f"out_{asii_3.index(c)}.png", outImg)
        del outImg
    
    print("".join(list(map((lambda x: x[0]), sorted(list(temp.items()), key=lambda x: x[1])))))
    
def translet(image:np.ndarray, fileout:str = "out.txt", *, _a:Iterable = asii_1) -> np.ndarray:
    # I = lambda x: (np.abs(np.min(x)) + x)/np.max(x)
    # temp_img = img2gray(image)
    # I2 = lambda x, e, fi: np.where(x>=e, 1, 1+np.tanh(fi*(x-e)))
    # outImg:np.ndarray = I2(I(img2DoG(temp_img, ((7,7),(9,9)), 1.4, 1)), 0.99, 1) 
    
    # temp:np.ndarray = img2blur(image, 7, 7)
    
    # GxImg:np.ndarray = img2filter2D(temp.dot(MeanMat2Red), a=Gx)
    # GyImg:np.ndarray = img2filter2D(temp.dot(MeanMat2Green), a=Gy)

    
    # GxImg_:np.ndarray = img2filter2D(temp.dot(MeanMat2Blue), a=Gx)
    # GyImg_:np.ndarray = img2filter2D(temp.dot(MeanMat2Blue), a=Gy)


    # _i = Img2quantize(((np.arctan2(GxImg_, GyImg_)*0.5/np.pi) + 0.5)*255, 5)
    
    # outImg:np.ndarray = GxImg+GyImg+np.sqrt(GxImg_**2 + GyImg_**2)
    
    outImg:np.ndarray = img2filter2D(image, a=Gx)

    return outImg 

def getImagis() -> np.ndarray :
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    if not len(fileNames):sys.exit(0)
    assert fileNames[-1].split('.')[-1] in ["png", "jpg", "jpeg", "webp"]
    queueImages:np.ndarray = imgbgr2rgb(cv2.imread(fileNames[-1]))
    return queueImages  

@staticmethod
def main(ars = None) -> None:
    queueImages:np.ndarray = getImagis()
    out_img:np.ndarray =  translet(queueImages)
    cv2.imwrite(FTEMP+f"out.png", out_img)




if __name__ == "__main__": 
    main()
    #input("Pleas press 'ENTER' to continue...")
    print("#END")
    

