
#region IMPORT
import math 
import sys
from typing import Any, Generator, Iterable, Sequence
import cv2
import cv2.data
import numpy as np  

from ThresholdMap import *  # type: ignore[reportMissingImports] 

import os
import os.path as PATH
from PIL import Image
from progress.bar import IncrementalBar, ChargingBar
import time

import matplotlib as mpl
import matplotlib.pyplot as plt  
from matplotlib.figure import Figure 
from matplotlib.axes import Axes 
import matplotlib.animation as animation

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
asii_3v = r" .',:`;\"i!I^lrvjcx<>Yft*JL?TuynozaksFVXehCqKUdpSZbAwGPgEOHmDQNR%&BWM#@$"
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_4 = r" .;coPO?@#"
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾


CWD = os.getcwd()
FRESU = CWD + "\\~resu\\"
FTEMP = CWD + "\\~temp\\"
S:float = 0.5
ESCCLEANER:str = "\033[0m"

I = lambda _i: (lambda x: x/np.max(x))((lambda x: x - np.min(x))(_i))

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

def img2ConsoleImg(image:np.ndarray, _a:str, w:int, h:int):
    global ESCCLEANER
    tempImg:np.ndarray = cv2.resize(image, (w, h))
    
    tempImgGray:np.ndarray = I_Img2Qtize(img2gray(tempImg)/255, len(_a))
    temp_f = np.vectorize(lambda x, y: _a[int(tempImgGray[x][y]//(1/(len(_a)-1))-1)])
    ss = np.fromfunction(temp_f, tempImgGray.shape, dtype=int)
    del tempImgGray, temp_f
    
    tempImg_:np.ndarray = I_Img2Qtize(tempImg/256, len(_a)) 

    tempImg_ = np.astype(I(tempImg_)*256, int)    
    
    _w, _h, _ = tempImg_.shape    
    
    @np.vectorize
    def f(x:int, y:int) -> str:
        r:str = str(tempImg_[x][y][0])
        g:str = str(tempImg_[x][y][1])
        b:str = str(tempImg_[x][y][2]) 
        return ";".join((r, g, b))
    
    t:np.ndarray = np.fromfunction(f, (_w, _h), dtype=int)
    _t:list = []
    for i in t.tolist():_t += i
    palette:list[str] = list(set(_t))
    
    del t, _t, f
    
    @np.vectorize
    def f(x:int, y:int):
        r:str = str(tempImg_[x][y][0])
        g:str = str(tempImg_[x][y][1])
        b:str = str(tempImg_[x][y][2]) 
        return palette.index(";".join((r, g, b)))
    tempImg_:np.ndarray = np.fromfunction(f, (_w, _h), dtype=int)
    
    @np.vectorize
    def temp_f(x:int, y:int):
        if y:
            if tempImg_[x][y] != tempImg_[x][y-1]:
                return "\033[38;2;" + palette[tempImg_[x][y]] + "m"
            else:
                return ""
        else:
            return "\033[38;2;" + palette[tempImg_[x][y]] + "m"
    
    ss_ = np.fromfunction(temp_f, (_w, _h), dtype=int)
    del tempImg_, temp_f, f
    
    temp_f = np.vectorize(lambda x, y: ss_[x][y] + ss[x][y] + ("", ESCCLEANER)[int(y >= (len(ss[x])-1))])
    strings:np.ndarray = np.fromfunction(temp_f, (_w, _h), dtype=int)
    
    del ss, ss_, _w, _h, _
    return strings

Img2quantize    = lambda x, h: (np.ceil(x/h) + 0.5)*h                                   
I_Img2quantize  = lambda x, k: np.ceil(x/k + 0.5)*k
I_Img2Qtize     = lambda x, k: np.ceil(x*(k-1) + 0.5)/(k-1)
img2filter2D    = lambda x, a=kernel2: cv2.filter2D(src=x, ddepth=-1, kernel=a)

#endregion IMG2

Sigmoid = lambda x: 1/(1 + np.e**(-x))

@np.vectorize
def GetMitemRGB(i:int, j:int, fi:int) -> float: 
    m:np.ndarray = ThresholdMap.Mat_8
    k:int = len(m)
    return  (m[i%k][j%k] - 0.5)
@np.vectorize
def GetMitem(i:int, j:int) -> float:
    m:np.ndarray = ThresholdMap.Mat_8
    k:int = len(m) 
    return  (m[(i)%k][(j)%k] - 0.5)

def img2Sharp(image:np.ndarray, sigma=1.0, strength=1.5, kernel_size=(5, 5)):
    
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return sharpened

def DoG(image:np.ndarray, kSize:int, rSize:float, teta:float = 1., *, sigmaX_1:float=0., sigmaX_2:float=0.) -> np.ndarray:
    assert kSize%2 and rSize > 0
    ksize_low = np.array((kSize, kSize))
    ksize_hight = np.int_(ksize_low*max(rSize, 1/rSize))
    return (1 + teta)*cv2.GaussianBlur(image, ksize_low, sigmaX_1) - teta*cv2.GaussianBlur(image, ksize_hight, sigmaX_2)

def translet(image:np.ndarray, fileout:str = "out.txt", *, _a:Iterable | str = asii_1, wh = tuple(os.get_terminal_size())) -> str:    
    w, h = wh
    # strings = img2ConsoleImg(image, _a, w, h)
    strings = img2ConsoleImg(image, _a, w, h)
    return "\n".join(["".join(s) for s in strings.tolist()])

def f_csShape4imgShape(imgShape:tuple[int, int, Any], csShape:tuple[int, int, Any]) ->  tuple[int, int]:
    imgW:int = imgShape[1]
    imgH:int = imgShape[0]
    imgTg:float = round(imgH/imgW, 2)
    csW:int = csShape[0]
    csH:int = csShape[1]
    # print(imgTg)
    return 2 * int(csH / imgTg), csH 

def getFileNames() -> list[str]:
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    while not len(fileNames):
        sel:str = input("Небыл(и) выбран(ы) файл(ы)! \nЕсли хотите вернутся к выбору файлов введите 'yes'[Y/n]: ").lower()
        if sel in "no0нет":
            sys.exit(0)
        fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    return fileNames

def getImagis(fileNames:list[str]):
    def sel():
        for fileName in fileNames:
            ind:int = fileName.index(fileName)
            if fileName.split('.')[-1] in ["png", "jpg", "jpeg", "webp"]:
                t:np.ndarray = img2CRev(cv2.imread(fileName))
                yield (ind, (t, None))
            else:
                cap:cv2.VideoCapture = cv2.VideoCapture(fileName)
                cap_len:int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_FPS:float = (cap.get(cv2.CAP_PROP_FPS))
                result:list[np.ndarray] = [img2CRev(cap.read()[1]) for _ in range(cap_len)]
                t:tuple = (result, cap_FPS)
                cap.release()
                yield (ind, t)
    return sel

def link(uri, label=None):
    if label is None: 
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)

def imgShow(img:np.ndarray) -> None:
    
    plt.imshow(img)
    plt.show()
    
    pass

def test(imgs:list[np.ndarray], fps):
    fig, ax = plt.subplots()    
    
    art = []
    
    for img in imgs:
        cont = ax.imshow(img)
        art.append([cont])
    ani = animation.ArtistAnimation(fig=fig, artists=art, interval= np.ceil(1000/fps))
    plt.show()


@staticmethod
def Main() -> None:
    
    fileNames:list[str] = getFileNames()
    gen_queueImages:Generator = getImagis(fileNames)
    
    for ind, image in gen_queueImages():
        Img, fps = image
        
        if type(Img) is list:
            imgShow(Img[0])
            _ti:list = []
            bar = IncrementalBar('Countdown', max = len(Img))
            imgSize =  np.array((Img[0].shape[1], Img[0].shape[0]), int)
            for img in Img:
                _img = np.fromfunction(GetMitemRGB, img.shape, dtype=int)
                temp:np.ndarray = I(I_Img2Qtize(I(img/256+S*_img), 2))
                
                temp = cv2.resize(temp, imgSize//2)
                _ti.append(cv2.resize(temp, imgSize))
                del temp
                bar.next()
            bar.finish()
            del bar
            
            test(_ti, fps)
        else:
            img:np.ndarray = img2Sharp(Img/256, 3.0, 5.5, (7,7))
            
            _img = np.fromfunction(GetMitemRGB, img.shape, dtype=int)

            imgShow(I(I_Img2Qtize(I(img+0.5*_img), 2)))
            
            
        #_img:np.ndarray = cv2.resize(img, (img.shape[0]//10, img.shape[1]//10))
        #imgShow(cv2.resize(_img, (img.shape[1], img.shape[0])))
        


if __name__ == "__main__": 
    #input("#START")
    #print("\033[H\033[J", end="")
    Main()
    #input("Pleas press 'ENTER' to continue...")
    #input("#END")
