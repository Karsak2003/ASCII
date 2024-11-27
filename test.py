from math import ceil
import sys
from typing import Any, Sequence
import cv2
import numpy as np  
import os

import matplotlib as mpl
import matplotlib.pyplot as plt  
from matplotlib.figure import Figure 
from matplotlib.axes import Axes 
import matplotlib.animation as animation

#from PySide6 import QtWidgets, QtCore, QtGui

from tkinter import filedialog as tk_filedialog



#__________________________________________________________________________________
asii = """$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'."""
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_1 = """`.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"""[::-1]
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#__________________________________________________________________________________
asii_2 = """$@B%8&WM#oahkbdpqwmZO0QLCJUYXzcvunxrjf1?+~ilI;:*^"',."""
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
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
img2DoG = (lambda x, a=(3, 3), b=(7, 7): img2blur(x, b[0], b[1])-img2blur(x, a[0], a[1]))

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
        
    
    
        

def ImgShow3(image_left, image_сentre, image_right) -> None:
    f, a = plt.subplots(1, 3)
    axs:list[Axes] = list(a)
    fig:Figure = f
    del a, f
    
    axs[0].imshow(image_left)
    axs[1].imshow(image_сentre)
    axs[2].imshow(image_right)

    plt.show()
    
def ImgShow(image) -> None:
    f, a = plt.subplots()
    axs:Axes = a
    fig:Figure = f
    del a, f
    axs.imshow(image)
    plt.show()
    
def ImgShowMany(images, n=1, m=1) -> None:
    fig, axs = plt.subplots(n, m) 
    
    if not (n-1 or m-1):
        axs.imshow(images)
        plt.show()
        return
    if n-1 and (not m-1):
        for i in range(n):
            axs[i].imshow(images[i])
        plt.show()
        return
    if (not n-1) and m-1:
        for j in range(m):
            axs[j].imshow(images[j])
        plt.show()
        return
    if n-1 and m-1:    
        for i in range(n):
            for j in range(m):
                axs[i][j].imshow(images[i][j])
        plt.show()
        return

def translet(image, fileout:str = "out.txt"):
    org_size = np.array([len(image), len(image[0])][::-1])
    new_size = org_size.copy()
    if org_size[0] > 1000:
        new_size = np.array([len(image)//23, len(image[0])//11][::-1])*5
    elif round(org_size[1]/org_size[0] - 11/23, 3) > 0:
        new_size[1] = int(new_size[1] * 11/23) 
        pass
    # while new_size[0] >= 1000:
    #     new_size//=2
    print(org_size, new_size)
    #600x840
    #11х23 

    out_img = cv2.resize((img2blur(image)), new_size)
    pater = asii_1 
    with open(FTEMP+fileout, mode="w+") as file:
        s = ""
        f1 = False
        f2 = False
        for i in img2not(img2gray(out_img)):
            for j in i:
                t =  j/256*len(pater)
                if 0.3 < t - int(t) < 0.7:
                    #0..1: 0, 0.25, 0.5, 0.75, 1
                    s+=pater[int(t) + int(f1)*((1, -1)[int(t)+1 >= len(pater)])]
                    f1=not f1
                elif t - int(t) <= 0.3: 
                    s+=pater[int(t)]
                else:
                    s+=pater[int(t)+1] 
            s+="\n"
        file.write(s)  
        #print(s)
    
    return out_img
    

@staticmethod
def main(ars = None) -> None:
    n_f = tk_filedialog.askopenfilename(initialdir=FRESU, initialfile="img(0).png")
    if n_f == "":sys.exit(0)
    images = []
    if n_f.split('.')[-1] in ["png", "jpg", "jpeg", "webp"]:
        images.append(cv2.imread(n_f))
    else:
        cap = cv2.VideoCapture(n_f)
        images = [imgbgr2rgb(cap.read()[1]) for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
        cap.release()
    
    for i in range(len(images)):
        # images[i] = imgbgr2rgb(images[i])
        images[i] = images[i]
    out_img = []
    for i in range(len(images)):
        out_img = translet(images[i], fileout=f"out({i}).txt")
        plt.imshow(images[i])

    cv2.imwrite(FTEMP+"out.png", img2not(out_img[:,:,0]))
    
    
    




if __name__ == "__main__": main()

    
    

