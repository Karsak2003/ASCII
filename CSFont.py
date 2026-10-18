
#region IMPORTS
import ctypes
from ctypes import wintypes
#endregion IMPORTS

#region CONST
# --- Константы Windows API ---
STD_OUTPUT_HANDLE = -11
LF_FACESIZE = 32  # Максимальная длина имени шрифта

TMPF_TRUETYPE = 0x04
FF_DONTCARE = 0x00
##endregion CONST

# --- Описание структур Windows API ---
class COORD(ctypes.Structure):
    """Структура для хранения координат/размеров (ширина и высота)."""
    _fields_ = [
        ("X", ctypes.c_short),
        ("Y", ctypes.c_short)
    ]

class CONSOLE_FONT_INFOEX(ctypes.Structure):
    """Структура, содержащая информацию о шрифте консоли."""
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("nFont", ctypes.c_ulong),
        ("dwFontSize", COORD),
        ("FontFamily", ctypes.c_int),
        ("FontWeight", ctypes.c_ulong),
        ("FaceName", (ctypes.c_wchar * LF_FACESIZE))
    ]

def get_current_console_font():
    """Получает информацию о текущем шрифте консоли."""
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    
    font_info = CONSOLE_FONT_INFOEX()
    font_info.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
    
    GetCurrentConsoleFontEx = kernel32.GetCurrentConsoleFontEx
    GetCurrentConsoleFontEx.argtypes = [
        wintypes.HANDLE, 
        wintypes.BOOL, 
        ctypes.POINTER(CONSOLE_FONT_INFOEX)
    ]
    GetCurrentConsoleFontEx.restype = wintypes.BOOL
    
    result = GetCurrentConsoleFontEx(handle, False, ctypes.byref(font_info))
    
    if result:
        return {
            'name': font_info.FaceName,
            'size': (font_info.dwFontSize.X, font_info.dwFontSize.Y),
            'weight': font_info.FontWeight
        }
    return None

def set_console_font(font_name="Consolas", font_size=(8, 16), font_weight=400):
    """
    Изменяет шрифт текущей консоли Windows.
    
    :param font_name: Название шрифта (например, "Consolas", "Lucida Console").
                      Поддерживаются только TrueType шрифты.
    :param font_size: Кортеж (ширина, высота) в пикселях.
    :param font_weight: Толщина шрифта (400 - обычный, 700 - жирный).
    """
    # Загружаем kernel32.dll
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    
    # 1. Получаем хендл стандартного вывода (окна консоли)
    handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    if handle == -1:  # INVALID_HANDLE_VALUE
        raise ctypes.WinError(ctypes.get_last_error())
        
    # 2. Заполняем структуру CONSOLE_FONT_INFOEX
    font_info = CONSOLE_FONT_INFOEX()
    font_info.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)  # Обязательно указываем размер структуры
    font_info.nFont = 0  # Для TrueType шрифтов всегда должно быть 0
    font_info.dwFontSize.X = font_size[0]
    font_info.dwFontSize.Y = font_size[1]
    #font_info.FontFamily = 0  # FF_DONTCARE (система сама определит семейство)
    font_info.FontFamily = TMPF_TRUETYPE  
    font_info.FontWeight = font_weight
    
    # 3. Копируем имя шрифта в массив FaceName
    # ctypes автоматически дополнит оставшиеся символы нулями (\0), если строка короче LF_FACESIZE
    #/ font_info.FaceName = (ctypes.c_wchar * LF_FACESIZE)(*font_name)
    face_name = (ctypes.c_wchar * LF_FACESIZE)()
    for i, char in enumerate(font_name):
        if i < LF_FACESIZE - 1:
            face_name[i] = char
        else:
            break
    font_info.FaceName = face_name

    # 4. Настраиваем сигнатуру функции SetCurrentConsoleFontEx
    SetCurrentConsoleFontEx = kernel32.SetCurrentConsoleFontEx
    SetCurrentConsoleFontEx.argtypes = [
        wintypes.HANDLE, 
        wintypes.BOOL, 
        ctypes.POINTER(CONSOLE_FONT_INFOEX)
    ]
    SetCurrentConsoleFontEx.restype = wintypes.BOOL
    
    # 5. Вызываем Windows API
    # Второй аргумент (bMaximumWindow) обычно False
    result = SetCurrentConsoleFontEx(handle, False, ctypes.byref(font_info))
    
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
        
    return True



# ==========================================
# Пример использования
# ==========================================
if __name__ == "__main__":
    try:
        import os
        print(f"Родительский процесс: {os.getppid()}")
        print("Текущий шрифт:", get_current_console_font())
        print("Меняем шрифт на Courier New ...")
        set_console_font(font_name="Courier New")
        print("Шрифт успешно изменен!")
        print("После изменения:", get_current_console_font())
        # Проверка вывода (например, для ASCII-графики из вашего проекта)
        print("\nПроверка вывода:")
        print("█ ▄ ▀ ▐ ▌ █ ▄ ▀ ▐ ▌ █ ▄ ▀ ▐ ▌")
        
    except Exception as e:
        print(f"Не удалось изменить шрифт: {e}")
        print("Примечание: Убедитесь, что скрипт запущен в реальном терминале (cmd/PowerShell), а не в IDE.")