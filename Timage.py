#from warnings import deprecated
#from tqdm import trange
#import random as rnd
#from typing import List

from PIL import Image
import numpy as np
from math import pi
pi = np.float32(pi)
from tqdm import trange

from IPython.display import display
from scipy.signal import convolve2d
from math import exp

from scipy.ndimage import median_filter

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
            self.__arr = array # TODO было np.array(self.__img) изза чего мы теряли мантиссы. сейчас необходимо чтоб входной массив был чб
            self.__dtype = self.__arr.dtype

    def __add__(self, other: "Timage") -> "Timage":
        if self.__arr.shape != other.__arr.shape:
            raise ValueError("Can't add images of different sizes.")
        out = self.__arr+other.__arr
        return Timage(array=out)
    
    def __repr__(self):
        display(self.__img)
        return ""
    
    def show(self, pallete=[0, 255], contrast_level: int = 0):
        np_pallete = np.array(pallete, dtype=np.float32)

        mean = np.mean(self.__arr)
        f = lambda x: (x - mean) / (1 - contrast_level) + mean
        contrasted = self.__arr.copy()

        m, n = len(self.__arr), len(self.__arr[0])
        for i in range(m):
            for j in range(n):
                contrasted[i][j] = min(255, max(0,   f(float(self.__arr[i][j]))  ))
            
        new_arr = np.multiply.outer(contrasted, np_pallete[1]/255) + np.multiply.outer(255-contrasted, np_pallete[0]/255)
        #return Image.fromarray(new_arr.astype('uint8')) #for saving
        Image.fromarray(new_arr.astype('uint8')).show() # Pillow can only generate images from uint8 and uint16 arrays, thats why astype('uint8') is necessary
        return Image.fromarray(new_arr.astype('uint8'))

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
        out = median_filter(self.__arr, size=(2*radius+1, 2*radius+1), mode='reflect')
        
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
        
    def defect_map(self, radius, stddev=1.5, contrast_level=0.997, direction=None, color=255):

        kernel = np.zeros((2*radius+1, 2*radius+1, 2), dtype=np.float32) # kernel[i][j] = [real, img] representing z = real + i * img
        gauss_func = lambda r: np.float32(np.exp(-r**2 / (2*stddev**2)) / np.float32(stddev * (2 * pi) ** 0.5))

        #
        # itializing kernel for convolution and sum s for normalization
        #

        s = np.float32(0)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                r = np.float32((i**2 + j**2)**0.5) # distance from center
                if r == 0: continue # for measuring change around current pixel we dont need current pixel itself
                g = gauss_func(r)
                s += g
                kernel[radius + i][radius + j][0] = (j / r) * g
                kernel[radius + i][radius + j][1] = (-i / r) * g # because i decreasing pixel is going up, but we need it to go down
        
        #normalizing kernel
        kernel /= s

        #
        # convolving self of kernel
        #

        d_map = np.zeros((len(self.array) - 2*radius, len(self.array[0]) - 2*radius, 2), dtype=np.float32)
        for i in trange(len(d_map), desc='Convolving...'):
            for j in range(len(d_map[0])):

                region = self.array[i:i+2*radius+1, j:j+2*radius+1]
                
                d_map[i][j] = np.sum(region[..., np.newaxis] * kernel, (0,1))
        
        #return d_map # if we want work with DefectMap

        #
        #replacing matrix of complex numbers with matrix of real numbers using magnitude or direction
        #
        
        if direction is None:
            directed = np.zeros((len(self.array) - 2*radius, len(self.array[0]) - 2*radius), dtype=np.float32)
            for i in range(len(d_map)):
                for j in range(len(d_map[0])):
                    directed[i][j] = (d_map[i][j][0]**2 + d_map[i][j][1]**2)**0.5
        else:
            directed = 128 + np.sum(d_map * direction, (2)) # directed[i][j] = dot_product(d_map[i][j], direction) i.e. float number
        
        #
        #contrast and coloring
        #

        mean = np.mean(directed)
        f = lambda x: (x - mean) / (1 - contrast_level) + mean
        contrasted = directed.copy()

        m, n = len(directed), len(directed[0])
        for i in range(m):
            for j in range(n):
                contrasted[i][j] = min(255, max(0,   f(float(directed[i][j]))  ))
        
        colored = np.multiply.outer(contrasted.copy(), np.array(color, dtype=np.float32)/255)
        
        #Image.fromarray(colored.astype('uint8')).show()
        return Image.fromarray(colored.astype('uint8'))