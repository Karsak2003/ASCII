
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

# import matplotlib as mpl
# import matplotlib.pyplot as plt  
# from matplotlib.figure import Figure 
# from matplotlib.axes import Axes 
# import matplotlib.animation as animation

from tkinter import filedialog as tk_filedialog
#endregion IMPORT

#region Const

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
#___________________________________________________________________________________________________
asii_5 = r' .:-=+*#%@'
asii_5v = """ .'^":;|,?+*#$%@"""
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

CWD = os.getcwd()
FRESU = CWD + "\\~resu\\"
FTEMP = CWD + "\\~temp\\"
# K:int = 8
S:float = 0.5
ESCCLEANER:str = "\033[0m"

# def Mat2N(n:int) -> np.ndarray:
#     assert not(n%2) and (type(n) is int)
#     if n > 2:
#         return np.kron(np.ones((2, 2)), Mat2N(n//2)) + np.kron(Mat2, np.ones((n//2, n//2)))/((n/2)**2)
#     else:
#         return Mat2

#region Matrix
kernel2 = np.ones((5, 5), np.float32)/25
Gx:np.ndarray               = np.array([[1,  0, -1],[2,  0, -2],[1,  0, -1]])
Gy:np.ndarray               = np.array([[1,  2, 1],[0,  0, 0],[-1,  -2, -1]])
MeanMat:np.ndarray          = np.array([1/3, 1/3, 1/3])
RedMat:np.ndarray           = np.array([[1, 0, 0],[0, 0, 0],[0, 0, 0]])
GreenMat:np.ndarray         = np.array([[0, 1, 0],[0, 0, 0],[0, 0, 0]])
BlueMat:np.ndarray          = np.array([[0, 0, 1],[0, 0, 0],[0, 0, 0]])
CRevMat:np.ndarray          = np.array([[0, 0, 1],[0, 1, 0],[1, 0, 0]])
MeanMat2Red:np.ndarray      = np.array([[1/3, 0, 0],[1/3, 0, 0],[1/3, 0, 0]])
MeanMat2Green:np.ndarray    = np.array([[0, 1/3, 0],[0, 1/3, 0],[0, 1/3, 0]])
MeanMat2Blue:np.ndarray     = np.array([[0, 0, 1/3],[0, 0, 1/3],[0, 0, 1/3]])
MeanMat2RGB:np.ndarray      = np.array([[1/3, 1/3, 1/3],[1/3, 1/3, 1/3],[1/3, 1/3, 1/3]])
MeanMat2:np.ndarray         = np.array([[1/3, 1/3, 1/3],[0, 0, 0],[0, 0, 0]])
#endregion Matrix
#endregion Const

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

img2filter2D    = lambda x, a=kernel2: cv2.filter2D(src=x, ddepth=-1, kernel=a)

def img2Angle(x:np.ndarray):
    GxImg_:np.ndarray = img2filter2D(x, a=Gx)
    GyImg_:np.ndarray = img2filter2D(x, a=Gy)
    contur:np.ndarray = (np.arctan(GyImg_/GxImg_)/np.pi + 1) * 0.5
    return contur 

def img2ConsoleImg(image:np.ndarray, _a:str, w:int, h:int) -> np.ndarray[str]:
    """Преобразует изображение в массив строк цветного ASCII-арта с ANSI-кодами.
    
    Масштабирует изображение до (w, h), квантует яркость для выбора символа из паллитры `_a`,
    а цвет пикселя кодирует через escape-последовательности `\\033[38;2;r;g;bm.`

    Args:
        image (np.ndarray): Исходное изображение в формате BGR (numpy.ndarray).
        _a (str): Строка символов палитры яркости (от тёмного к светлому).
        w (int): Целевая ширина вывода в символах.
        h (int): Целевая высота вывода в символах.

    Returns:
        np.ndarray[str]: Массив строк, где каждый элемент — строка ANSI-последовательности.
    """
    global ESCCLEANER
    tempImg:np.ndarray = cv2.resize(image, (w, h))
    
    tempImgGray:np.ndarray = I_Img2quantize(img2gray(tempImg)/255, 1/(len(_a)-1))
    
    bins:np.ndarray[float] = np.linspace(0, 1, len(_a))
    indices:np.ndarray = np.digitize(tempImgGray, bins) - 1
    indices:np.ndarray = np.clip(indices, 0, len(_a)-1)
    
    ss:np.ndarray = np.array(list(_a))[indices]
    
    del tempImgGray, bins, indices
    
    #Квантование цветов    
    tempImg_:np.ndarray = I_Img2quantize(tempImg/255, 1/(len(_a)-1)) 

    #Нормализация данных
    tempImg_ -= tempImg_.min()
    tempImg_ /= tempImg_.max()
    tempImg_ =  tempImg_ * 255
    #Преобразование в целочисленный тип
    tempImg_ = np.astype(tempImg_, int)    
    
    #
    _w, _h, _ = tempImg_.shape    
    
    paletteFond:list[str] = list(set(getPalette(tempImg_)))
        
    palette_dict = {color: idx for idx, color in enumerate(paletteFond)}
    tempImg_indices:np.ndarray = np.zeros((_w, _h), dtype=int)
    
    for color, idx in palette_dict.items():
        
        r_val, g_val, b_val, = map(int, color.split(';')) 
        mask = (tempImg_[:, :, 0] == r_val) & (tempImg_[:, :, 1] == g_val) & (tempImg_[:, :, 2] == b_val)
        tempImg_indices[mask] = idx
        
    tempImg_:np.ndarray = tempImg_indices
    
    # Сравниваем каждый пиксель с предыдущим по оси Y (горизонтальная строка)
    shifted:np.ndarray = np.roll(tempImg_, 1, axis=1)
    mask = np.ones((_w, _h), dtype=bool)
    mask[:, 1:] = (tempImg_[:, 1:] != shifted[:, 1:])  # True, если цвет изменился или это первый столбец
    
    # Генерируем массив ANSI-кодов для всех возможных индексов
    ansi_codes = np.array(["\033[38;2;" + color + "m" for color in paletteFond], dtype=object)
    
    # Применяем коды только там, где mask == True, иначе пустая строка
    ss_:np.ndarray = np.where(mask, ansi_codes[tempImg_], "")
    
    del tempImg_, palette_dict, tempImg_indices, mask, shifted, ansi_codes
    
    # Объединяем ANSI-коды и символы
    combined:np.ndarray = ss_ + ss
    
    # Добавляем ESCCLEANER только к последнему элементу каждой строки (по оси 1)
    last_col_mask = np.zeros((_w, _h), dtype=bool)
    last_col_mask[:, -1] = True
    combined[last_col_mask] += ESCCLEANER
    
    del ss, ss_, _w, _h, _, last_col_mask
    return combined

Img2quantize    = lambda x, h: (np.ceil(x/h) + 0.5)*h                                   
I_Img2quantize  = lambda x, k: np.ceil(x/k + 0.5)*k
#endregion IMG2

Sigmoid = lambda x: 1/(1 + np.e**(-x))

def SelectContour(image:np.ndarray, size, *, x = 0.75) -> tuple[str, np.ndarray]:
    new_size:np.ndarray[int] = np.array((size[0], size[1]), dtype=int)
    
    temp:np.ndarray = I_Img2quantize(img2gray(image)/255, 1/(2**8))
    
    temp_contur:np.ndarray = np.where(DoG(temp, 5, 7/5, 0.95) >=x, 1.0, 0.)
    temp_contur-=temp_contur.min()
    temp_contur/=temp_contur.max()
    
    contur:np.ndarray = I_Img2quantize(img2Angle(temp_contur), 1/8)
    contur:np.ndarray = (2*contur-1) * 180
    contur:np.ndarray = np.where(contur < 0, contur + 180, contur)
    
    out_contur:np.ndarray = cv2.resize(np.abs(contur//45), new_size)
    stroca:np.ndarray[str] = np.empty(out_contur.shape, str)

    del contur, temp_contur, temp
    
    # @np.vectorize
    # def translation2symbols(i:int, j:int):
    #     x:int = out_contur[i][j]
    #     return ["\\", "|", "/", "_"][int(x)] if not np.isnan(x) else " "
    
    # stroca = np.fromfunction(translation2symbols, out_contur.shape, dtype=int)
    
    symbols = np.array(["\\", "|", "/", "_"], dtype=object)
    stroca = np.full(out_contur.shape, " ", dtype=object)
    
    valid_mask = ~np.isnan(out_contur) 
    x_valid = out_contur[valid_mask].astype(int)
    stroca[valid_mask] = symbols[x_valid]
    
    return "\n".join(["".join(s) for s in stroca]), out_contur

def DoG(image:np.ndarray, kSize:int, rSize:float, teta:float = 1., *, sigmaX_1:float=0., sigmaX_2:float=0.) -> np.ndarray:
    assert kSize%2 and rSize > 0
    ksize_low = np.array((kSize, kSize))
    ksize_hight = np.int_(ksize_low*max(rSize, 1/rSize))
    return (1 + teta)*cv2.GaussianBlur(image, ksize_low, sigmaX_1) - teta*cv2.GaussianBlur(image, ksize_hight, sigmaX_2)

#region Gets
@np.vectorize
def GetMitemRGB(i:int, j:int, fi:int) -> float: 
    m:np.ndarray = ThresholdMap.Mat_8
    k:np.ndarray = len(m)
    return  S*(m[i%k][j%k] - 0.5)

@np.vectorize
def GetMitem(i:int, j:int) -> float: 
    m:np.ndarray = ThresholdMap.Mat_8
    k:np.ndarray = len(m)
    return  S*(m[(i)%k][(j)%k] - 0.5)

"""
# Функции GetMitem и GetMitemRGB (если они используются)
Если вы генерируете карту порогов через эти функции, замените их вызовы на одну векторизованную операцию тайлинга (размножения матрицы):
Вместо поэлементного вызова:
```
    # Предположим, вам нужен массив размером (H, W)
    k = len(ThresholdMap.Mat_8)
    tile_h = (H + k - 1) // k
    tile_w = (W + k - 1) // k

    # Мгновенно создаем нужную матрицу без циклов
    threshold_map = S * (np.tile(ThresholdMap.Mat_8, (tile_h, tile_w))[:H, :W] - 0.5)
```
"""

def getPalette(tempImg_:np.ndarray[int], *, scale:int=1) -> list[str]:
    """
    Извлекает уникальные RGB-значения из изображения в формате "R;G;B".
    
    Args:
        tempImg_: массив изображения `shape(H, W, 3)` с int значениями
        
    Returns:
        list[str]: плоский список строк вида "R;G;B"
    """    
    r:np.ndarray = (tempImg_[:, :, 0]//scale).astype(str)
    g:np.ndarray = (tempImg_[:, :, 1]//scale).astype(str)
    b:np.ndarray = (tempImg_[:, :, 2]//scale).astype(str)
    
    t:np.ndarray = np.char.add( np.char.add( np.char.add( np.char.add(r, ';'), g),  ';'), b)

    return t.flatten().tolist()

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
#endregion Gets

def translet(image:np.ndarray, fileout:str = "out.txt", *, _a:Iterable | str = asii_1, wh = tuple(os.get_terminal_size())) -> str:
    """Вызов преобразования изображения и упаковка выходного массива/списка строк в одну единную строку

    Args:
        image (np.ndarray): Изображение
        fileout (str, optional): Имя выходного файла (не реализуеться, использовалось как заглушка). Defaults to "out.txt".
        _a (Iterable | str, optional): Паллитра ASCII-символов. Defaults to asii_1.
        wh (tuple[int, int], optional): Ширина и высота консоли. Defaults to tuple(os.get_terminal_size()).

    Returns:
        str: Строка с ASCII-art
    """
    w, h = wh
    # strings = img2ConsoleImg(image, _a, w, h)
    strings = img2ConsoleImg(image, _a, w, h)
    return "\n".join(["".join(s) for s in strings.tolist()])

def csShape4imgShape(imgShape:tuple[int, int, Any], csShape:tuple[int, int, Any]) ->  tuple[int, int]:
    """Рассчитывает размеры области консоли для вывода ASCII-арта с сохранением пропорций изображения.

    Компенсирует прямоугольную форму символов терминала (множитель 2 для ширины)
    и адаптирует ширину вывода под заданную высоту консоли.

    Args:
        imgShape (tuple[int, int, Any]): Кортеж размеров исходного изображения (высота, ширина, каналы).
        csShape (tuple[int, int, Any]): Кортеж целевых размеров консоли (ширина, высота).

    Returns:
        tuple[int, int]: Кортеж (ширина, высота) для корректного отображения изображения в консоли.
    """
    imgW:int = imgShape[1]
    imgH:int = imgShape[0]
    imgTg:float = round(imgH/imgW, 2)
    csW:int = csShape[0]
    csH:int = csShape[1]
    # print(imgTg)
    return 2 * int(csH / imgTg), csH 

def link(uri, label=None):
    if label is None: 
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)


@staticmethod
def Main(ars = None, *, _a = asii_3v) -> None:
    
    fileNames:list[str] = getFileNames()
    #gen_queueImages:Generator[tuple[int, tuple[np.ndarray, None]], tuple[int, tuple[list[np.ndarray], float]]] = getImagis(fileNames)()

    wh = tuple(os.get_terminal_size())
    
    dTFrame:int = 1/60#second
    duration:int = 5
    print(wh)
    input("Получение данных завершино.\nНажмите 'ENTER' для продолжения...")
    # print(f"\033[8;{wh[1]+5};{wh[0]+5}")
    
    
    for ind, image in getImagis(fileNames)():
        # _ts_orign:str = f"исходник: <a href=\"{fileNames[ind]}\">{fileNames[ind].split("/")[-1]}</a>"
        _ts_orign:str = f"исходник: {link(fileNames[ind], fileNames[ind].split("/")[-1])}"
        img, fps = image
        
        if type(img) is list:
                        
            images:list[np.ndarray] = img
            lenght_ing:int = len(images) 
            ss:list[str] = []
            dTFrame:int = 1/fps
            
            wh = csShape4imgShape(tuple(images[0].shape), wh)
            
            bar = IncrementalBar('Countdown', max = lenght_ing)
            for i in range(lenght_ing):
                bar.next()
                ss.append(translet(images[i], fileout=f"out({ind})({i}).txt", _a=_a, wh = wh))
                # ss.append(SelectContour(images[i], size=wh)[0])
            bar.finish()
            del bar
            
            _ts_orign += (lambda hw: f":{hw[0]}x{hw[1]}|{hw[0]/hw[1]};")(images[0].shape)
            
            print("\033[H\033[J", end="")
            
            i_counter:int = 0
            while True:
                i:int = i_counter % lenght_ing
                temp_ts:tuple = (i_counter, i_counter // lenght_ing, (duration / (dTFrame * lenght_ing)))
                _ts:str = f"#{temp_ts[0]} цик; \t {i+1}/{lenght_ing} кадр; \t {temp_ts[1]}//{temp_ts[2]} = {temp_ts[1]//temp_ts[2]}"
                
                print(_ts_orign + f"\tРазрешение: {wh[1]}x{wh[0]}|{wh[1]/wh[0]};" + "\n"+ f"FPS:{1/dTFrame}\t" + _ts + "\n" + ss[i], flush=True)

                with open(FTEMP+f"test({i}).ans", "w+") as f: print( ss[i], file=f, flush=True)
                
                if temp_ts[1] >= temp_ts[2]:                    
                    input("Нажмите 'ENTER' для продолжения...")
                    print("\033[H\033[J", end="")
                    break
                i_counter+=1
                time.sleep(dTFrame)
                print("\033[H\033[3J", end="", flush=True)
        else:
            print("\033[H\033[J", end="")
            wh = csShape4imgShape(tuple(img.shape), wh)
            _ts_orign += (lambda wh: f":{wh[1]}x{wh[0]}|{wh[0]/wh[1]};")(img.shape)
            
            print(_ts_orign + f"\tРазрешение: {wh[0]}x{wh[1]}|{wh[1]/wh[0]};")
            _s:str = translet(img, fileout=f"out({ind}).txt", _a=_a, wh=wh)
            print(_s, flush=True)
            with open(FTEMP+f"test_.ans", "w+") as f: print( _s, file=f, flush=True)
            
            input("Нажмите 'ENTER' для продолжения...")
            # time.sleep(1)
            print("\033[H\033[J", end="", flush=True)



if __name__ == "__main__": 
    input("#START")
    print("\033[H\033[J", end="")
    Main(_a=asii_3v)
    print("\033[H\033[3J", end="", flush=True)
    input("Pleas press 'ENTER' to continue...")

    
    
    

