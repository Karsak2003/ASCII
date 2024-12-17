
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
S:float = 1.

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
img2gray        = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
img2bgr         = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
img2rgb         = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
img2CRev        = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
img2not         = lambda x: cv2.bitwise_not(x)
img2and         = lambda x, y: cv2.bitwise_and(x, y) 
img2or          = lambda x, y: cv2.bitwise_or(x, y)
img2xor         = lambda x, y: cv2.bitwise_xor(x, y)
img2blur        = lambda x, a=7, b=7, s=0: cv2.GaussianBlur(x, (a, b), s)
img2cont        = lambda x: cv2.Canny(x, 100, 160)
img2DoG         = lambda x, a=((3, 3),(7, 7)), s=0., t=1.0: (1+t)*img2blur(x, a[0][0], a[0][1], s)-t*img2blur(x, a[1][0], a[1][1], s)
img2Laplacian   = lambda x: cv2.Laplacian(x, cv2.CV_64F)

def img2Angle(x:np.ndarray):
    GxImg_:np.ndarray = img2filter2D(x, a=Gx)
    GyImg_:np.ndarray = img2filter2D(x, a=Gy)
    contur:np.ndarray = (np.arctan(GyImg_/GxImg_)/np.pi + 1) * 0.5
    return contur

def img2ConsoleImg(image, _a, w, h):
    tempImg:np.ndarray = cv2.resize(image, (w, h))
    
    tempImgGray:np.ndarray = I_Img2quantize(img2gray(tempImg)/255, 1/(len(_a)-1))
    temp_f = np.vectorize(lambda x, y: _a[int(tempImgGray[x][y]//(1/(len(_a)-1))-1)])
    ss = np.fromfunction(temp_f, tempImgGray.shape, dtype=int)
    del tempImgGray, temp_f
    
    tempImg_:np.ndarray = np.round((tempImg)/255)
    _w, _h, _ = tempImg_.shape
    temp_f = np.vectorize(lambda x, y: int("".join(list(map(str, list(map(int,tempImg_[x-1][y-1]))))), 2))
    ss_ = np.fromfunction(temp_f, (_w, _h), dtype=int)
    del tempImg_, temp_f, 
    
    temp_f = np.vectorize(lambda x, y: f"\033[{30+ss_[x][y]}m"+ ss[x][y] + "\033[0m")
    strings:np.ndarray = np.fromfunction(temp_f, (_w, _h), dtype=int)
    
    del ss, ss_, _w, _h, _
    return strings


Img2quantize    = lambda x, h: (np.ceil(x/h) + 0.5)*h
I_Img2quantize  = lambda x, k: np.ceil(x/k + 0.5)*k
img2filter2D    = lambda x, a=kernel2: cv2.filter2D(src=x, ddepth=-1, kernel=a)
#endregion IMG2

Sigmoid = lambda x: 1/(1 + np.e**(-x))

@np.vectorize
def GetMitemRGB(i:int, j:int, fi:int) -> float: 
    return  S*(M[i%K][j%K] - 0.5)
@np.vectorize
def GetMitem(i:int, j:int) -> float: 
    return  S*(M[(i)%K][(j)%K] - 0.5)

"""
def _where(x, x2T, x2F) -> Any:
    def f(f_temp):
        def _f_(*atr):
            assert len(atr) in [3, 5]
            if len(atr) == 3:
                image, kSize, rSize, = atr
                return np.where(f_temp(image, kSize, rSize) >= x, x2T, x2F)
            else:
                image, kSize, rSize, sigmaX_1, sigmaX_2 = atr
                return np.where(f_temp(image, kSize, rSize, sigmaX_1=sigmaX_1, sigmaX_2=sigmaX_2) >= x, x2T, x2F)
            
        return _f_
    return f

@_where(0.01, 1, 0)
"""

def DoG(image:np.ndarray, kSize:int, rSize:float, teta:float = 1., *, sigmaX_1:float=0., sigmaX_2:float=0.) -> np.ndarray:
    assert kSize%2 and rSize > 0
    ksize_low = np.array((kSize, kSize))
    ksize_hight = np.int_(ksize_low*max(rSize, 1/rSize))
    return (1 + teta)*cv2.GaussianBlur(image, ksize_low, sigmaX_1) - teta*cv2.GaussianBlur(image, ksize_hight, sigmaX_2)

def ImgShow(image) -> None:
    f, a = plt.subplots()
    axs:Axes = a
    fig:Figure = f
    del a, f
    axs.imshow(image)
    plt.show()


def __txt():
    temp = dict()
    
    for c in asii_3:
        outImg:np.ndarray = np.zeros((27, 25, 3), np.uint8)
        cv2.putText(outImg, c, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255))
        temp[c] = outImg.sum()
        #ImgShow(outImg)
        cv2.imwrite(FTEMP+f"out_{asii_3.index(c)}.png", outImg)
        del outImg
    
    print("".join(list(map((lambda x: x[0]), sorted(list(temp.items()), key=lambda x: x[1])))))

def translet(image:np.ndarray, fileout:str = "out.txt", *, _a:Iterable = asii_1) -> None:    
    w, h = tuple(os.get_terminal_size())
    strings = img2ConsoleImg(image, _a, w, h)
    return "\n".join(["".join(s) for s in strings.tolist()])
    


    
def getImagis() -> np.ndarray :
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    if not len(fileNames):sys.exit(0)
    assert fileNames[-1].split('.')[-1] in ["png", "jpg", "jpeg", "webp"]
    # queueImages:np.ndarray = img2CRev(cv2.imread(fileNames[-1]))
    queueImages:np.ndarray = cv2.imread(fileNames[-1])
    return queueImages  

def getImagis4test() -> list[np.ndarray | list[np.ndarray]]:
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    if not len(fileNames):sys.exit(0)
    F_not_gif:bool = False
    queueImages:list[np.ndarray | list[np.ndarray]] = []
    for fileName in fileNames:
        F_not_gif = fileName.split('.')[-1] in ["png", "jpg", "jpeg", "webp"]
        if F_not_gif:
            queueImages.append(img2CRev(cv2.imread(fileName)))
        else:
            cap = cv2.VideoCapture(fileName)
            # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            queueImages.append([img2CRev(cap.read()[1]) for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))])
            cap.release()
    return queueImages


def packing2GIF(lenFrames:int, frames:list[Image.Image] = [], fileout:str="out") ->  None:
    for i in range(lenFrames):
        with Image.open(fileout+f"({i}).png") as frame: 
            frames.append(frame.copy())
        os.remove(fileout+f"({i}).png")
    frames[0].save(
                fileout + '.gif',
                save_all=True,
                append_images=frames[1:],  # Срез который игнорирует первый кадр.
                optimize=True,
                duration=100,
                loop=0
            )
    frames.clear()


@staticmethod
def main(ars = None) -> None:
    queueImages:np.ndarray  = getImagis()
    out_img:np.ndarray      =  translet(queueImages, _a=asii_3)
    cv2.imwrite(FTEMP+f"out.png", out_img)

@staticmethod
def main4test(ars = None) -> None:
    queueImages:list[np.ndarray | list[np.ndarray]] = getImagis4test()
    _a = asii_3
    for ind in range(len(queueImages)):
        if type(queueImages[ind]) is list:
            images:list = queueImages[ind]
            ss:list[str] = []
            for i in range(len(images)):
                ss.append(translet(images[i], fileout=f"out({ind})({i}).txt", _a=_a))
            
            i_counter:int = 0
            while True:
                i = i_counter % len(images)
                print(ss[i])
                if (i_counter // len(images))//10:
                    input("Нажмите 'ENTER' для продолжения...")
                    print("\033[F\033[J", end="")
                    break
                i_counter+=1
                time.sleep(0.25)
                print("\033[H\033[J", end="")
        else:translet(queueImages[ind], fileout=f"out({ind}).txt", _a=_a)


if __name__ == "__main__": 
    input("#START")
    print("\033[H\033[J", end="")
    main4test()
    #input("Pleas press 'ENTER' to continue...")
    input("#END")
    
SGA:dict[str, str] = {
"a": "ᔑ",
"b": "ʖ",
"c": "ᓵ",
"d": "↸",
"e": "ᒷ",
"f": "⎓",
"g": "⊣",
"h": "⍑",
"я": "╎",
"j": "⋮",
"k": "ꖌ",
"l": "ꖎ",
"m": "ᒲ",
"n": "リ",
"o": "𝙹",
"p": "⇅",
"q": "ᑑ",
"r": "∷",
"s": "ᓭ",
"t": "ℸ",
"u": "⚍",
"v": "⍊",
"w": "∴",
"x": "/",
"y": "|",
"z": "⨅",}
"""Standard Galactic Alphabet"""