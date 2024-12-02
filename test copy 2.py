
#region IMPORT
import math 
import sys
from typing import Any, Iterable, Sequence
import cv2
import cv2.data
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
#// asii_3 = r" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
asii_3 = r" .',:`;\"i!I^lr1vjcx<>Yft*JL?T7uynozaksFVXeh3Cq2KUdp4SZbA0w5GPg9EOH6mDQNR8%&BWM#@$"
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

CWD = os.getcwd()
FRESU = CWD + "\\~resu\\"
FTEMP = CWD + "\\~temp\\"
K:int = 8
S:float = 0.01

def Mat2N(n:int) -> np.ndarray:
    assert not(n%2) and (type(n) is int)
    if n > 2:
        return np.kron(np.ones((2, 2)), Mat2N(n//2)) + np.kron(Mat2, np.ones((n//2, n//2)))/((n/2)**2)
    else:
        return Mat2

#region Matrix
kernel2 = np.ones((5, 5), np.float32)/25
Gx:np.ndarray               = np.array([[1,  0, -1],[2,  0, -2],[1,  0, -1]])
Gy:np.ndarray               = np.array([[1,  2, 1],[0,  0, 0],[-1,  -2, -1]])
MeanMat:np.ndarray          = np.array([1/3, 1/3, 1/3])
RedMat:np.ndarray           = np.array([[1, 0, 0],[0, 0, 0],[0, 0, 0]])
GreenMat:np.ndarray         = np.array([[0, 1, 0],[0, 0, 0],[0, 0, 0]])
BlueMat:np.ndarray          = np.array([[0, 0, 1],[0, 0, 0],[0, 0, 0]])
MeanMat2Red:np.ndarray      = np.array([[1/3, 0, 0],[1/3, 0, 0],[1/3, 0, 0]])
MeanMat2Green:np.ndarray    = np.array([[0, 1/3, 0],[0, 1/3, 0],[0, 1/3, 0]])
MeanMat2Blue:np.ndarray     = np.array([[0, 0, 1/3],[0, 0, 1/3],[0, 0, 1/3]])
MeanMat2RGB:np.ndarray      = np.array([[1/3, 1/3, 1/3],[1/3, 1/3, 1/3],[1/3, 1/3, 1/3]])
MeanMat2:np.ndarray         = np.array([[1/3, 1/3, 1/3],[0, 0, 0],[0, 0, 0]])

Mat2:np.ndarray             = np.array(
    [[0, 2],
     [3, 4]]
)/4
M:np.ndarray    = Mat2N(K)
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
img2DoG = lambda x, a=((3, 3),(7, 7)), s=0., t=1.0: (1+t)*img2blur(x, a[0][0], a[0][1], s)-t*img2blur(x, a[1][0], a[1][1], s)
img2Laplacian = lambda x: cv2.Laplacian(x, cv2.CV_64F)

def img2Angle(x:np.ndarray):
    GxImg_:np.ndarray = img2filter2D(x, a=Gx)
    GyImg_:np.ndarray = img2filter2D(x, a=Gy)
    contur:np.ndarray = (np.arctan(GyImg_/GxImg_)/np.pi + 1) * 0.5
    return contur

Img2quantize = lambda x, h: (np.ceil(x/h) + 0.5)*h
I_Img2quantize = lambda x, k: np.ceil(x/k + 0.5)*k
img2filter2D = lambda x, a=kernel2: cv2.filter2D(src=x, ddepth=-1, kernel=a)
#endregion IMG2

Sigmoid = lambda x: 1/(1 + np.e**(-x))


@np.vectorize
def GetMitemRGB(i:int, j:int, fi:int) -> float: 
    return  S*(M[i%K][j%K] - 0.5)
@np.vectorize
def GetMitem(i:int, j:int) -> float: 

    return  S*(M[(i)%K][(j)%K] - 0.5)



def ImgShow(image) -> None:
    f, a = plt.subplots()
    axs:Axes = a
    fig:Figure = f
    del a, f
    axs.imshow(image)
    plt.show()

def ImgShow3(image_left, image_сentre, image_right) -> None:
    f, a = plt.subplots(1, 3)
    axs:list[Axes] = list(a)
    fig:Figure = f
    del a, f
    
    axs[0].imshow(image_left)
    axs[1].imshow(image_сentre)
    axs[2].imshow(image_right)

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
    org_size:np.ndarray[int] = np.array([len(image), len(image[0])][::-1])
    new_size = np.array([998, int((org_size[1]*998//org_size[0])*(11/23))])
    temp:np.ndarray = cv2.resize(image, new_size)
    
    temp = I_Img2quantize(img2gray(temp), (len(_a)))
    
    contur:np.ndarray = I_Img2quantize(img2Angle(temp), 1/8)
    contur = (2*contur-1) * 180
    contur = np.where(contur < 0, contur + 180, contur)

    out_contur:np.ndarray = np.abs(contur//45)
    stroca:np.ndarray[str] = np.empty(out_contur.shape, str)
    
    with open(FTEMP+"1_"+fileout, mode="w+") as file:
        @np.vectorize
        def translation2symbols(i:int, j:int):
            x:int = out_contur[i][j]
            return ["\\", "|", "/", "_"][int(x)] if not np.isnan(x) else " "
        stroca = np.fromfunction(translation2symbols, out_contur.shape, dtype=int)
        print("\n".join(["".join(s) for s in stroca]), file=file, flush=True)
    
    ImgShow(contur)
    cv2.imwrite(FTEMP+f"_out.png", contur*255)
    
    del temp, contur, out_contur
    
    temp:np.ndarray = img2gray(cv2.resize(image, new_size))/255
    temp += np.fromfunction(GetMitem, temp.shape, dtype=int)
    temp -= temp.min()
    temp /= temp.max()

    outImg:np.ndarray = I_Img2quantize(temp, 1/(len(_a)-1)) // (1/(len(_a)-1)) - 1
    #// with open(FTEMP+"0_"+fileout, mode="w+") as file: print(str(set(outImg[np.logical_not(np.isnan(outImg))].tolist())), file=file, flush=True)

    with open(FTEMP+"0_"+fileout, mode="w+") as file:
        @np.vectorize
        def translation2symbols(i:int, j:int):
            return _a[int(outImg[i][j])]
        stroca_ = np.fromfunction(translation2symbols, outImg.shape, dtype=int)
        print("\n".join(["".join(s) for s in stroca_]), file=file, flush=True)
    
    with open(FTEMP+"2_"+fileout, mode="w+") as file:
        @np.vectorize
        def translation2symbols(i:int, j:int):
            return _a[int(outImg[i][j])] if stroca[i][j] == " " else stroca[i][j]
        stroca_ = np.fromfunction(translation2symbols, outImg.shape, dtype=int)
        print("\n".join(["".join(s) for s in stroca_]), file=file, flush=True)
    
    ImgShow(outImg)
    return outImg 

def getImagis() -> np.ndarray :
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    if not len(fileNames):sys.exit(0)
    assert fileNames[-1].split('.')[-1] in ["png", "jpg", "jpeg", "webp"]
    # queueImages:np.ndarray = img2CRev(cv2.imread(fileNames[-1]))
    queueImages:np.ndarray = cv2.imread(fileNames[-1])
    return queueImages  

@staticmethod
def main(ars = None) -> None:
    queueImages:np.ndarray = getImagis()
    out_img:np.ndarray =  translet(queueImages, _a=asii_3)
    cv2.imwrite(FTEMP+f"out.png", out_img)




if __name__ == "__main__": 
    main()
    #input("Pleas press 'ENTER' to continue...")
    print("#END")
    

