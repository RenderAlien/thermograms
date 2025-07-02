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
        ###t = time()
        np.seterr(over='raise') # Сломать всё при переполнении
        if ((image is None) == (array is None)) or (
            (image is None) != (dtype is None)):
            raise ValueError(
                "Exactly one of 'image' or 'arr' must be provided to initialize Timage."
            )
        elif image is not None: #Сразу переводим в серый
            self.__img = image.convert("L")
            self.__arr = np.array(self.__img, dtype=dtype)
            self.__dtype = dtype
        else:
            self.__img = Image.fromarray(array).convert("L")
            self.__arr = array # TODO было np.array(self.__img) изза чего мы теряли мантиссы. сейчас необходимо чтоб входной массив был чб
            self.__dtype = self.__arr.dtype.type

        ###print('__init__ ' + str(time()-t))

    def __add__(self, other: "Timage") -> "Timage":
        if self.__arr.shape != other.__arr.shape:
            raise ValueError("Can't add images of different sizes.")
        out = self.__arr+other.__arr
        return Timage(array=out)
    
    def __repr__(self):
        display(self.__img)
        return ""
    
    def __getitem__(self, index):

        try:
            x, y, _type = index
        except:
            x, y = index
            _type = None
        
        coord_type = int|float
        if _type is None and (  not isinstance(x, coord_type) or not isinstance(y, coord_type)  ):
            raise KeyError(
                f"Inappropriate type for coordinates of thermogram, {x, y, _type} cannot be parameters."
            )

        int_type = int|None
        if _type == 'int' or isinstance(x, int_type) and isinstance(y, int_type):
            if 0 <= x < len(self.__arr) and 0 <= y < len(self.__arr[0]):
                return self.__arr[x, y]
            else:
                return self.dtype(0)
        else:
            if -2 <= x < len(self.__arr) and -2 <= y < len(self.__arr[0]): # -2 is for smooth edges
                return self.__bilinear_interpolate(x, y)
            else:
                return self.dtype(0)
            
    def __bilinear_interpolate(self, x, y):
        # https://en.m.wikipedia.org/wiki/Bilinear_interpolation
        vx0 = self[int(x), int(y), 'int'] + (self[int(x)+1, int(y), 'int'] - self[int(x), int(y), 'int']) * (x % 1)
        vx1 = self[int(x), int(y)+1, 'int'] + (self[int(x)+1, int(y)+1, 'int'] - self[int(x), int(y)+1, 'int']) * (x % 1)
        value = vx0 + (vx1 - vx0) * (y % 1)
        return value
    
    def __bicubic_interpolate(self, i, j):
        # https://en.m.wikipedia.org/wiki/Bicubic_interpolation
        int_x, int_y = int(i), int(j)

        def f(x, y, der=None):
            if der is None: return self[int_x+x, int_y+y, 'int']
            elif der == 'x': return (self[int_x+x+1, int_y+y, 'int'] - self[int_x+x-1, int_y+y, 'int'])/2
            elif der == 'y': return (self[int_x+x, int_y+y+1, 'int'] - self[int_x+x, int_y+y-1, 'int'])/2
            elif der == 'xy': return (f(x, y+1, 'x') - f(x, y-1, 'x')) / 2

        a0 = np.array([[1,0,0,0],
                        [0,0,1,0],
                        [-3,3,-2,-1],
                        [2,-2,1,1]], dtype=self.dtype)
        a1 = np.array([[f(0,0), f(0,1), f(0,0,'y'), f(0,1,'y')],
                       [f(1,0), f(1,1), f(1,0,'y'), f(1,1,'y')],
                       [f(0,0,'x'), f(0,1,'x'), f(0,0,'xy'), f(0,1,'xy')],
                       [f(1,0,'x'), f(1,1,'x'), f(1,0,'xy'), f(1,1,'xy')]], dtype=self.dtype)
        a2 = np.array([[1,0,-3,2],
                       [0,0,3,-2],
                       [0,1,-2,1],
                       [0,0,-1,1]], dtype=self.dtype)
        
        A = a0 @ a1 @ a2

        i -= int_x
        j -= int_y

        a0 = np.array([1, i, i**2, i**3], dtype=self.dtype)
        a2 = np.array([[1],[j],[j**2],[j**3]], dtype=self.dtype)

        p = a0 @ A @ a2
        return p[0]
    
    def show(self, pallete=[0, 255], contrast_level: int = 0):
        np_pallete = np.array(pallete, dtype=np.float32)

        mean = np.mean(self.__arr)
        f = lambda x: (x - mean) / (1 - contrast_level) + mean
        
        f = lambda x: (x - mean) / (1 - contrast_level) + mean

        contrasted = f(self.__arr)
        contrasted[contrasted < 0] = 0
        contrasted[contrasted > 255] = 255
            
        new_arr = np.multiply.outer(contrasted, np_pallete[1]/255) + np.multiply.outer(255-contrasted, np_pallete[0]/255)
        #return Image.fromarray(new_arr.astype('uint8')) #for saving
        #Image.fromarray(new_arr.astype('uint8')).show() # Pillow can only generate images from uint8 and uint16 arrays, thats why astype('uint8') is necessary
        return Image.fromarray(new_arr.astype('uint8'))

    @property
    def image(self) -> Image:
        return self.__img.copy()

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
        
    def defect_map(self, radius, stddev=1.5, contrast_level=0.997, direction=None, pallete=[0,255]):

        np_pallete = np.array(pallete, dtype=np.float32) # list -> np.array

        kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.complex64) # kernel[i][j] is representing z = real + i * img
        gauss_func = lambda r: np.exp(-r**2 / (2*stddev**2)) / np.float32(stddev * (2 * pi) ** 0.5)

        #
        # initializing kernel for convolution and sum s for normalization
        #

        s = np.float32(0)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                r = np.float32((i**2 + j**2)**0.5) # distance from center
                if r == 0: continue # for measuring change around current pixel we dont need current pixel itself
                g = gauss_func(r)
                s += g
                kernel[radius + i][radius + j] = np.complex64(complex(j / r, -i / r)) * g # because i decreasing pixel is going up, but we need it to go down
        
        #normalizing kernel
        kernel /= s

        #
        # convolving self of kernel
        #

        d_map = convolve2d(self.array, kernel, mode='same', boundary='symm')
        
        #return d_map # if we want to work with DefectMap

        #
        #replacing matrix of complex numbers with matrix of real numbers using magnitude or direction
        #
        
        if direction is None:
            directed = np.abs(d_map) # magnitude of each complex number
        else:
            directed = 128 - np.real(d_map * np.conjugate(direction)) # directed[i][j] = 128 - dot_product(d_map[i][j], direction) i.e. float number, idk why '-' instead of '+' TODO
        # explanation: d_map[i][j] * dir* = (  real(d_map[i][j]) * real(dir) + img(d_map[i][j]) * img(dir)  ) + i * (...)
        # thus, real(d_map[i][j] * dir*) = dot_product(d_map[i][j], dir) if d_map[i][j] and dir are vectors

        #
        #contrast and coloring
        #

        mean = np.mean(directed)
        #f = lambda x: (x - mean) / (1 - contrast_level) + mean

        contrasted = (directed - mean) / (1 - contrast_level) + mean
        contrasted[contrasted < 0] = 0
        contrasted[contrasted > 255] = 255
        
        colored = np.multiply.outer(contrasted, np_pallete[1]/255) + np.multiply.outer(255-contrasted, np_pallete[0]/255)
        
        #Image.fromarray(colored.astype('uint8')).show()
        return Image.fromarray(colored.astype('uint8'))



