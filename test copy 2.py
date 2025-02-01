
#region IMPORT
import math 
import sys
from typing import Any, Generator, Iterable, Sequence
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
asii_3v = r" .',:`;\"i!I^lrvjcx<>Yft*JL?TuynozaksFVXehCqKUdpSZbAwGPgEOHmDQNR%&BWM#@$"
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
    
    I_ = lambda R, G, B: "\033[38;2;" + ";".join((R, G, B)) + "m"
    _Cleaner:str = "\033[0m"
    
    temp_f = np.vectorize(lambda x, y: f"\033[{30+ss_[x][y]}m"+ ss[x][y] + _Cleaner)
    strings:np.ndarray = np.fromfunction(temp_f, (_w, _h), dtype=int)
    
    del ss, ss_, _w, _h, _
    return strings

def _img2ConsoleImg(image:np.ndarray, _a:str, w:int, h:int):

    tempImg:np.ndarray = cv2.resize(image, (w, h))
    _Cleaner:str = "\033[0m"
    
    tempImgGray:np.ndarray = I_Img2quantize(img2gray(tempImg)/255, 1/(len(_a)-1))
    temp_f = np.vectorize(lambda x, y: _a[int(tempImgGray[x][y]//(1/(len(_a)-1))-1)])
    ss = np.fromfunction(temp_f, tempImgGray.shape, dtype=int)
    del tempImgGray, temp_f
    
    tempImg_:np.ndarray = I_Img2quantize(tempImg/255, 1/(len(_a)-1)) 

    tempImg_ -= tempImg_.min()
    tempImg_ /= tempImg_.max()
    tempImg_ = tempImg_ * 255
    
    print(type(tempImg_[0][0]))
    tempImg_ = np.astype(tempImg_, int)
    print(type(tempImg_[0][0]))


    
    
    _w, _h, _ = tempImg_.shape    
    
    temp_f = np.vectorize(lambda x, y: (lambda R, G, B: "\033[38;2;" + ";".join((R, G, B)) + "m")(
        str(tempImg_[x][y][0]),
        str(tempImg_[x][y][1]),
        str(tempImg_[x][y][2])))
    ss_ = np.fromfunction(temp_f, (_w, _h), dtype=int)
    del tempImg_, temp_f, 
    
    temp_f = np.vectorize(lambda x, y: ss_[x][y] + ss[x][y] + ("", _Cleaner)[y >= (len(ss[x])-1)])
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

def SelectContour(image:np.ndarray, size, *, x = 0.75) -> tuple[str, np.ndarray]:
    new_size:np.ndarray[int] = np.array((size[0], size[1]))
    
    temp:np.ndarray = I_Img2quantize(img2gray(image)/255, 1/(2**8))
    
    temp_contur:np.ndarray = np.where(DoG(temp, 5, 7/5, 0.95) >=x, 1.0, 0.)
    temp_contur-=temp_contur.min()
    temp_contur/=temp_contur.max()
    
    contur:np.ndarray = I_Img2quantize(img2Angle(temp_contur), 1/8)
    contur:np.ndarray = (2*contur-1) * 180
    contur:np.ndarray = np.where(contur < 0, contur + 180, contur)
    
    out_contur:np.ndarray = cv2.resize(np.abs(contur//45), new_size)
    stroca:np.ndarray[str] = np.empty(out_contur.shape, str)

    del contur, temp_contur, temp, x
    
    @np.vectorize
    def translation2symbols(i:int, j:int):
        x:int = out_contur[i][j]
        return ["\\", "|", "/", "_"][int(x)] if not np.isnan(x) else " "
    stroca = np.fromfunction(translation2symbols, out_contur.shape, dtype=int)
    return "\n".join(["".join(s) for s in stroca]), out_contur

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

def translet(image:np.ndarray, fileout:str = "out.txt", *, _a:Iterable = asii_1, wh = tuple(os.get_terminal_size())) -> None:    
    w, h = wh
    # strings = img2ConsoleImg(image, _a, w, h)
    strings = _img2ConsoleImg(image, _a, w, h)
    return "\n".join(["".join(s) for s in strings.tolist()])
    


    
def _getImagis() -> np.ndarray :
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    if not len(fileNames):sys.exit(0)
    assert fileNames[-1].split('.')[-1] in ["png", "jpg", "jpeg", "webp"]
    # queueImages:np.ndarray = img2CRev(cv2.imread(fileNames[-1]))
    queueImages:np.ndarray = cv2.imread(fileNames[-1])
    return queueImages  

def getImagis4test() -> list[np.ndarray | list[np.ndarray]]:
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    while not len(fileNames):
        fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
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
    return queueImages, fileNames

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

def link(uri, label=None):
    if label is None: 
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)


@staticmethod
def main(ars = None) -> None:
    queueImages:np.ndarray  = _getImagis()
    out_img:np.ndarray      = translet(queueImages, _a=asii_3)
    cv2.imwrite(FTEMP+f"out.png", out_img)

@staticmethod
def main4test(ars = None) -> None:
    qi, fn = getImagis4test()
    
    fileNames:list[str] = list(fn)
    queueImages:list[np.ndarray | list[np.ndarray]] = qi
    
    del  qi, fn
    
    _a = asii_3v
    wh = tuple(os.get_terminal_size())
    dTFrame:int = 1/60#second
    duration:int = 5
    
    input("Получение данных завершино.\nНажмите 'ENTER' для продолжения...")
    
    for ind in range(len(queueImages)):
        # _ts_orign:str = f"исходник: <a href=\"{fileNames[ind]}\">{fileNames[ind].split("/")[-1]}</a>"
        _ts_orign:str = f"исходник: {link(fileNames[ind], fileNames[ind].split("/")[-1])};"
        if type(queueImages[ind]) is list:
            
            images:list = queueImages[ind]
            lenght_ing:int = len(images) 
            ss:list[str] = []
            dTFrame:int = max(dTFrame, 2/(lenght_ing*3))
            
            bar = IncrementalBar('Countdown', max = lenght_ing)
            for i in range(lenght_ing):
                bar.next()
                ss.append(translet(images[i], fileout=f"out({ind})({i}).txt", _a=_a, wh = wh))
                # ss.append(SelectContour(images[i], size=wh)[0])
            bar.finish()
            del bar
            
            print("\033[H\033[J", end="")
            
            i_counter:int = 0
            while True:
                i:int = i_counter % lenght_ing
                temp_ts:tuple = (i_counter, i_counter // lenght_ing, (duration / (dTFrame * lenght_ing)))
                _ts:str = f"#{temp_ts[0]} цик; \t {i+1}/{lenght_ing} кадр; \t {temp_ts[1]}//{temp_ts[2]} = {temp_ts[1]//temp_ts[2]}"
                
                print(_ts_orign + "\t" + _ts + "\n" + ss[i], flush=True)
                
                if temp_ts[1] >= temp_ts[2]:                    
                    input("Нажмите 'ENTER' для продолжения...")
                    print("\033[H\033[J", end="")
                    break
                i_counter+=1
                time.sleep(dTFrame)
                print("\033[H\033[3J", end="", flush=True)
        else:
            print(_ts_orign)
            print(translet(queueImages[ind], fileout=f"out({ind}).txt", _a=_a), flush=True)
            input("Нажмите 'ENTER' для продолжения...")
            # time.sleep(1)
            print("\033[H\033[J", end="", flush=True)


@staticmethod
def Main(ars = None) -> None:
    
    fileNames:list[str] = getFileNames()
    gen_queueImages:Generator[tuple[int, tuple[np.ndarray, None]], tuple[int, tuple[list[np.ndarray], float]]] = getImagis(fileNames)()

    _a = asii_3v
    wh = tuple(os.get_terminal_size())
    dTFrame:int = 1/60#second
    duration:int = 5
    
    input("Получение данных завершино.\nНажмите 'ENTER' для продолжения...")
    # print(f"\033[8;{wh[1]+5};{wh[0]+5}")
    
    
    for ind, image in gen_queueImages:
        # _ts_orign:str = f"исходник: <a href=\"{fileNames[ind]}\">{fileNames[ind].split("/")[-1]}</a>"
        _ts_orign:str = f"исходник: {link(fileNames[ind], fileNames[ind].split("/")[-1])};"
        img, fps = image
        if type(img) is list:
            
            images:list = img
            lenght_ing:int = len(images) 
            ss:list[str] = []
            dTFrame:int = 1/fps
            
            bar = IncrementalBar('Countdown', max = lenght_ing)
            for i in range(lenght_ing):
                bar.next()
                ss.append(translet(images[i], fileout=f"out({ind})({i}).txt", _a=_a, wh = wh))
                # ss.append(SelectContour(images[i], size=wh)[0])
            bar.finish()
            del bar
            
            print("\033[H\033[J", end="")
            
            i_counter:int = 0
            while True:
                i:int = i_counter % lenght_ing
                temp_ts:tuple = (i_counter, i_counter // lenght_ing, (duration / (dTFrame * lenght_ing)))
                _ts:str = f"#{temp_ts[0]} цик; \t {i+1}/{lenght_ing} кадр; \t {temp_ts[1]}//{temp_ts[2]} = {temp_ts[1]//temp_ts[2]}"
                
                print(_ts_orign + "\n"+ f"FPS:{1/dTFrame}\t" + _ts + "\n" + ss[i], flush=True)
                
                with open(FTEMP+"2_"+f"test({ind})({i}).txt", mode="w+") as file:
                    file.write(_ts_orign + "\n"+ f"FPS:{1/dTFrame}\t" + _ts + "\n" + ss[i])
                    
                                
                if temp_ts[1] >= temp_ts[2]:                    
                    input("Нажмите 'ENTER' для продолжения...")
                    print("\033[H\033[J", end="")
                    break
                i_counter+=1
                time.sleep(dTFrame)
                print("\033[H\033[3J", end="", flush=True)
        else:
            print("\033[H\033[J", end="")
            print(_ts_orign)
            print(translet(img, fileout=f"out({ind}).txt", _a=_a), flush=True)
            input("Нажмите 'ENTER' для продолжения...")
            # time.sleep(1)
            print("\033[H\033[J", end="", flush=True)


if __name__ == "__main__": 
    input("#START")
    print("\033[H\033[J", end="")
    Main()
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