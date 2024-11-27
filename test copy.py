#region IMPORT
from math import ceil
import sys
from typing import Any, Sequence
import cv2
import numpy as np  
import os
from PIL import Image
from progress.bar import IncrementalBar, ChargingBar
import time

# import matplotlib as mpl
# import matplotlib.pyplot as plt  
# from matplotlib.figure import Figure 
# from matplotlib.axes import Axes 
# import matplotlib.animation as animation

# from PySide6 import QtWidgets, QtCore, QtGui

from tkinter import filedialog as tk_filedialog
#endregion IMPORT


#__________________________________________________________________________________
asii = """$@B%8&WM*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`'. """
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_1 = """ `.-':,^=;><+!rc*z?sLTvJ7FiCfI31tluneoZ5Yxjya2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"""[::-1]
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_2 = """$@B%8&WM#oahkbdpqwmZO0QLCJUYXzcvunxrjf1?+~ilI;:*^"',."""
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
asii_3 = r" .',:`;_\"-i!\\I^lr1vjcx(/<>Yft|)*JL?T+7uynoz}={aksFVXe~[h3C]q2KUdp4SZbA0w5GPg9EOH6mDQNR8%&BWM#@$"
asii_4 = r" .',:`;\"i!I^lr1vjcx<>Yft*JL?T7uynozaksFVXeh3Cq2KUdp4SZbA0w5GPg9EOH6mDQNR8%&BWM#@$"

CWD = os.getcwd()
FRESU = CWD + "\\~resu\\"
FTEMP = CWD + "\\~temp\\"

img2gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
img2bgr  = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
img2rgb  = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
imgbgr2rgb  = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
img2not  = lambda x: cv2.bitwise_not(x)
img2and  = lambda x, y: cv2.bitwise_and(x, y) 
img2or   = lambda x, y: cv2.bitwise_or(x, y)
img2xor  = lambda x, y: cv2.bitwise_xor(x, y)
img2blur = lambda x, a=7, b=7: cv2.GaussianBlur(x, (a, b), 0)
img2cont = lambda x: cv2.Canny(x, 100, 160)
img2Sobel = lambda x, dx = 1, dy = 1: cv2.Sobel(x, ddepth=cv2.CV_64F, dx=dx, dy=dy, ksize=3, scale=0.1)
img2SobelComb = lambda x: cv2.addWeighted(img2Sobel(x, dy=0), 0.5, img2Sobel(x, dx=0), 0.5, 0)
img2Apletude = lambda x: np.sqrt(img2Sobel(x, dy=0)**2 + img2Sobel(x, dx=0)**2)
img2Angel = lambda x: np.arctan2(img2Sobel(x, dy=0), img2Sobel(x, dx=0))/np.pi * 0.5 + 0.5
img2DoG = (lambda x, a=(3, 3), b=(7, 7): img2blur(x, b[0], b[1])-img2blur(x, a[0], a[1]))
img2Laplacian = lambda x: cv2.Laplacian(x, cv2.CV_64F)

kernel2 = np.ones((5, 5), np.float32)/25
img2filter2D = lambda x, a=kernel2: cv2.filter2D(src=x, ddepth=-1, kernel=a)

# class Ui_MainWindow(Ui_Form):
#     def setupUi(self, Form, gif):
#         super().setupUi(Form)
#         self.movie = QtGui.QMovie(gif)
#         cap = cv2.VideoCapture(FRESU+"img(11).gif")
#         cap.read()
#         self.label.setMovie(self.movie)
#         self.movie.start()

def grayImg2quantize(grayImg:np.ndarray, k:int) -> np.ndarray:
    img = grayImg.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            img[i][j]=(img[i][j]//k)*k
    return img
def Img2quantize(grayImg:np.ndarray, k:int) -> np.ndarray:
    img = grayImg.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            for y in range(3):
                img[i][j][y]=(img[i][j][y]//k)*k
    return img

I_Img2quantize = lambda x, k: np.ceil(x/k + 0.5)*k

def translet(image:np.ndarray, fileout:str = "out.txt", *, _a:float = asii_1):
    org_size:np.ndarray[int] = np.array([len(image), len(image[0])][::-1])
    new_size = np.array([998, int((org_size[1]*998//org_size[0])*(11/23))])
    # print(org_size)
    # print(new_size)
    #600x840
    #11х23 
    
    out_img = img2gray(np.uint8(I_Img2quantize(image, len(_a))))/255*len(_a)
    #I = lambda x: img2Angel(img2and(img2DoG(x, a=(1, 1), b=(3, 3)), img2cont(x)))

    contur = img2Angel(out_img)
    out_contur = cv2.resize(contur, new_size)
    
    # out_contur = cv2.resize(I_Img2quantize(contur*255, 4), new_size)
    # cv2.imwrite(FTEMP+f"out_5.png", I_Img2quantize(contur*255, 4))
    
    out_img = cv2.resize(out_img, new_size)
    
    
    
    with open(FTEMP+"0_"+fileout, mode="w+") as file:
        f1 = False
        stroca = [list(" "*len(out_img[0])) for _ in range(len(out_img))]
        for i in range(len(out_img)):
            for j in range(len(out_img[i])):
                if 1/3 < (out_img[i][j] % 1) < 2/3:
                    #0..1: 0, 0.25, 0.5, 0.75, 1
                    stroca[i][j] = _a[int(out_img[i][j]) + int(f1)*((1, 0)[int(out_img[i][j]+1 >= len(_a))])]
                    f1=not f1
                elif out_img[i][j]%1 <= 1/3:  stroca[i][j] = _a[int(out_img[i][j])]
                else: stroca[i][j] = _a[int(out_img[i][j])+1]
                #stroca[i][j] = (stroca[i][j], "_", "/", "\\", "|")[int(out_contur[i][j])]
                # if out_contur[i][j] < 0.125: stroca[i][j]="_"
                # elif 0.125 <= out_contur[i][j] < 0.375: stroca[i][j] = "/"
                # elif 0.375 <= out_contur[i][j] < 0.625: pass
                # elif 0.625 <= out_contur[i][j] < 0.875: stroca[i][j] ="\\"
                # elif 0.875 <= out_contur[i][j]: stroca[i][j] ="|"
        
        file.write("\n".join(["".join(s) for s in stroca]))  

    with open(FTEMP+"1_"+fileout, mode="w+") as file:
        f1 = False
        stroca = [list(" "*len(out_img[0])) for _ in range(len(out_img))]
        for i in range(len(out_img)):
            for j in range(len(out_img[i])):
                # if 1/3 < (out_img[i][j] % 1) < 2/3:
                #     #0..1: 0, 0.25, 0.5, 0.75, 1
                #     stroca[i][j] = _a[int(out_img[i][j]) + int(f1)*((1, 0)[int(out_img[i][j]+1 >= len(_a))])]
                #     f1=not f1
                # elif out_img[i][j]%1 <= 1/3:  stroca[i][j] = _a[int(out_img[i][j])]
                # else: stroca[i][j] = _a[int(out_img[i][j])+1]
                #stroca[i][j] = (stroca[i][j], "_", "/", "\\", "|")[int(out_contur[i][j])]
                if out_contur[i][j] < 0.125: stroca[i][j]="_"
                elif 0.125 <= out_contur[i][j] < 0.375: stroca[i][j] = "/"
                elif 0.375 <= out_contur[i][j] < 0.625: stroca[i][j] = " "
                elif 0.625 <= out_contur[i][j] < 0.875: stroca[i][j] ="\\"
                elif 0.875 <= out_contur[i][j]: stroca[i][j] ="|"
        
        file.write("\n".join(["".join(s) for s in stroca]))  
    
    with open(FTEMP+fileout, mode="w+") as file:
        f1 = False
        stroca = [list(" "*len(out_img[0])) for _ in range(len(out_img))]
        for i in range(len(out_img)):
            for j in range(len(out_img[i])):
                if 1/3 < (out_img[i][j] % 1) < 2/3:
                    #0..1: 0, 0.25, 0.5, 0.75, 1
                    stroca[i][j] = _a[int(out_img[i][j]) + int(f1)*((1, 0)[int(out_img[i][j]+1 >= len(_a))])]
                    f1=not f1
                elif out_img[i][j]%1 <= 1/3:  stroca[i][j] = _a[int(out_img[i][j])]
                else: stroca[i][j] = _a[int(out_img[i][j])+1]
                #stroca[i][j] = (stroca[i][j], "_", "/", "\\", "|")[int(out_contur[i][j])]
                if out_contur[i][j] < 0.125: stroca[i][j]="_"
                elif 0.125 <= out_contur[i][j] < 0.375: stroca[i][j] = "/"
                elif 0.375 <= out_contur[i][j] < 0.625: pass
                elif 0.625 <= out_contur[i][j] < 0.875: stroca[i][j] ="\\"
                elif 0.875 <= out_contur[i][j]: stroca[i][j] ="|"
        
        file.write("\n".join(["".join(s) for s in stroca]))  
    
    out_contur*=255
    
    return out_img, out_contur

def getImagis() -> list[np.ndarray | list[np.ndarray]]:
    fileNames:tuple[str] = tk_filedialog.askopenfilenames(initialdir=FRESU, initialfile="img(0).png")
    if not len(fileNames):sys.exit(0)
    F_not_gif:bool = False
    queueImages:list[np.ndarray | list[np.ndarray]] = []
    for fileName in fileNames:
        F_not_gif = fileName.split('.')[-1] in ["png", "jpg", "jpeg", "webp"]
        if F_not_gif:
            queueImages.append(imgbgr2rgb(cv2.imread(fileName)))
        else:
            cap = cv2.VideoCapture(fileName)
            # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            queueImages.append([imgbgr2rgb(cap.read()[1]) for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))])
            cap.release()
    return queueImages

def packing2GIF(lenFrames:int, frames:list[Image.Image] = [], fileout:str="out", *, bar:IncrementalBar = None) ->  None:
    startIndBar = bar.index
    startMaxBar = bar.max
    bar.index = 0
    bar.max = lenFrames+1
    for i in range(lenFrames):
        with Image.open(fileout+f"({i}).png") as frame: 
            frames.append(frame.copy())
        os.remove(fileout+f"({i}).png")
        bar.next()
    frames[0].save(
                fileout + '.gif',
                save_all=True,
                append_images=frames[1:],  # Срез который игнорирует первый кадр.
                optimize=True,
                duration=100,
                loop=0
            )
    frames.clear()
    bar.next()
    bar.index = startIndBar
    bar.max = startMaxBar
    pass

@staticmethod
def main(ars = None) -> None:
    queueImages:list[np.ndarray | list[np.ndarray]] = getImagis()
    
    out_img = []
    frames:list[Image.Image] = []
    _a = asii_4
    bar = IncrementalBar('Countdown', max = len(queueImages))
    for ind in range(len(queueImages)):
        if type(queueImages[ind]) is list:
            images:list = queueImages[ind]
            bar.index = 0
            bar.max = len(images)+2
            for i in range(len(images)):
                bar.next()
                out_img, out_contur = translet(images[i], fileout=f"out({ind})({i}).txt", _a=_a)
                cv2.imwrite(FTEMP+f"out({ind})({i}).png", out_img)
                cv2.imwrite(FTEMP+f"_out({ind})({i}).png", out_contur)
            packing2GIF(len(images), frames, FTEMP+f"out({ind})", bar=bar)
            bar.next()
            packing2GIF(len(images), frames, FTEMP+f"_out({ind})", bar=bar)
            bar.next()
            bar.max = len(queueImages)
            bar.index = ind
        else:
            out_img, out_contur = translet(queueImages[ind], fileout=f"out({ind}).txt", _a=_a)
            cv2.imwrite(FTEMP+f"out({ind}).png", out_img)
            cv2.imwrite(FTEMP+f"_out({ind}).png", out_contur)
            
        bar.next()
    bar.finish()



if __name__ == "__main__": 
    main()
    input("Pleas press 'ENTER' to continue...")

    

