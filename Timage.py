from warnings import deprecated
from tqdm import trange
import random as rnd
from typing import List

from PIL import Image
import numpy as np

from IPython.display import display
from scipy.signal import convolve2d, medfilt2d

class Timage:
    def __init__(self, *, image: Image = None, array: np.ndarray = None, dtype: np.dtype = None) -> None:
        """_summary_

        Args:
            image (_type_, optional): Image representing the array data. Defaults to None.
            array (np.ndarray, optional): 2D numpy array representing the image data. Defaults to None.

        Raises:
            ValueError: Exactly one of 'image' or 'arr' must be provided to initialize Timage.
        """
        np.seterr(over='raise') # Сломать всё при переполнении
        if ((image is None) == (array is None)) or (
            (image is None) != (dtype is None)):
            raise ValueError(
                "Exactly one of 'image' or 'arr' must be provided to initialize Timage."
            )
        elif image is not None: #Сразу переводим в серый
            self.__img = image.convert("L")
            self.__arr = np.array(self.__img)
            self.__dtype = dtype
        else:
            self.__img = Image.fromarray(array).convert("L")
            self.__arr = np.array(self.__img)
            self.__dtype = self.__arr.dtype

    def __add__(self, other: "Timage") -> "Timage":
        if self.__arr.shape != other.__arr.shape:
            raise ValueError("Can't add images of different sizes.")
        out = self.__arr+other.__arr
        return Timage(array=out)
    
    def __repr__(self):
        display(self.__img)
        return ""
    
    def show(self):
        self.__img.show()

    @property
    def image(self) -> Image:
        return self.__img.copy()

    #TODO удалить
    @property
    def array(self) -> np.ndarray:
        return self.__arr.copy()
    
    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    def median_blur(self, radius=3) -> "Timage":
        # x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        out = medfilt2d(self.__arr, radius*2+1)
        
        return Timage(array=out)

    def gaussian_blur(self, blur=1, radius=3) -> "Timage":
        x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        G = np.exp(-(x**2 + y**2) / (2 * blur**2)) / (2 * np.pi * blur**2)
        G /= np.sum(G) #kernel normalization
        
        out = convolve2d(self.__arr, G, mode='same', boundary='symm').astype(self.dtype)
        return Timage(array=out)

    def salt_and_pepper_noise(self, intensity=0.1) -> "Timage":
        out = self.__arr.copy()
        rnd = np.random.random(size=out.shape)
        out[rnd <= intensity / 2] = 0
        out[(rnd > intensity / 2) & (rnd <= intensity)] = 255

        return Timage(array=out)
    
    def gaussian_noise(self, mean=0, stddev=32) -> "Timage":
        rnd =np.random.normal(mean, stddev, size = self.__arr.shape)
        out = self.__arr + np.minimum(255-self.__arr, rnd)
        return Timage(array=out)
        
if __name__ == "__main__":
    ...
    
    