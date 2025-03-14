
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

CWD = os.getcwd()
FRESU = CWD + "\\~resu\\"
FTEMP = CWD + "\\~temp\\"
K:int = 8
S:float = 1.
ESCCLEANER:str = "\033[0m"

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

Img2quantize    = lambda x, h: (np.ceil(x/h) + 0.5)*h                                   
I_Img2quantize  = lambda x, k: np.ceil(x/k + 0.5)*k
img2filter2D    = lambda x, a=kernel2: cv2.filter2D(src=x, ddepth=-1, kernel=a)

def img2ConsoleImg(image:np.ndarray, _a:str, w:int, h:int):
    pass

#endregion IMG2

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
