#from warnings import deprecated
#from tqdm import trange
#import random as rnd
#from typing import List

from PIL import Image
import numpy as np
from math import pi, acos, cos, sin
np_pi = np.float32(pi)
from tqdm import trange

from IPython.display import display
from scipy.signal import convolve2d
from math import exp

from scipy.ndimage import median_filter

CAM_KI = [1, -0.0000015, 4.8*0.000001**2] # coefficients for camera distortion correction

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

        if isinstance(index, slice):
            return Timage(array=self.array[index])

        if len(index) == 3:
            x, y, _type = index
        elif len(index) == 2:
            if isinstance(index[0], slice) and isinstance(index[1], slice): 
                return Timage(array=self.array[index])
            x, y = index
            _type = None
        
        if _type is None and (  not isinstance(x, int|float) or not isinstance(y, int|float)  ):
            raise KeyError(
                f"Inappropriate type for coordinates of thermogram, {x, y, _type} cannot be parameters."
            )

        if _type == 'int' or isinstance(x, int) and isinstance(y, int):
            if 0 <= x < len(self.__arr) and 0 <= y < len(self.__arr[0]):
                return self.__arr[x, y]
            else:
                return self.dtype(0)
        else:
            if 0 <= x < len(self.__arr) and 0 <= y < len(self.__arr[0]):
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
    
    #def __bififthpower_interpolate(self, i, j):
    #    int_i, int_j = int(i), int(j)
    #    J = np.array([[jk**k for k in range(6)] for jk in range(-2, 4)], dtype=self.dtype)
    #    T = np.array([[self[ii, jj, 'int'] for jj in range(int_j-2, int_j+4)] for ii in range(int_i-2, int_i+4)], dtype=self.dtype)
    #    A_i = np.linalg.inv(J) @ T
    #    J_x = np.array([(j-int_j)**k for k in range(6)], dtype=self.dtype)
    #    T_x = J_x @ A_i
    #    I = np.array([[ii**k for k in range(6)] for ii in range(-2, 4)], dtype=self.dtype)
    #    A_jx = np.linalg.inv(I) @ np.array([[T_x[jj]] for jj in range(6)])
    #    I_x = np.array([(i-int_i)**k for k in range(6)], dtype=self.dtype)
#
    #    t_xx = I_x @ A_jx
    #    return t_xx[0]
    
    def show(self, pallete=[[0,0,0],[255,255,255]], contrast_level: int = 0):
        np_pallete = np.array(pallete, dtype=np.float32)

        mean = np.mean(self.__arr[self.__arr > 0]) # [self.__arr > 0] is to avoid overlighting picture because of zeros
        f = lambda x: (x - mean) / (1 - contrast_level) + mean

        contrasted = f(self.__arr)
        contrasted[contrasted < 0] = 0
        contrasted[contrasted > 255] = 255
            
        colored = np.multiply.outer(contrasted, np_pallete[1]/255) + np.multiply.outer(255-contrasted, np_pallete[0]/255)
        colored[self.__arr == 0] = self.dtype(0) if np_pallete[0].shape==() else np.zeros(np_pallete[0].shape, dtype=self.dtype)
        #Image.fromarray(new_arr.astype('uint8')).show() # Pillow can only generate images from uint8 and uint16 arrays, thats why astype('uint8') is necessary
        return Image.fromarray(colored.astype('uint8'))

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
        
    def defect_map(self, radius, stddev=1.5, contrast_level=0.997, direction=None, pallete=[[0,0,0],[255,255,255]]):

        np_pallete = np.array(pallete, dtype=np.float32) # list -> np.array

        kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.complex64) # kernel[i][j] is representing z = real + i * img
        gauss_func = lambda r: np.exp(-r**2 / (2*stddev**2)) / np.float32(stddev * (2 * np_pi) ** 0.5)

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

    def resized(self, shape):
        new = np.zeros(shape, dtype=self.dtype)
        ki = (self.__arr.shape[0] - 1) / (shape[0] - 1)
        kj = (self.__arr.shape[1] - 1) / (shape[1] - 1)

        for i in range(shape[0]):
            for j in range(shape[1]):
                new[i,j] = self[i*ki, j*kj]
        
        return Timage(array=new)
    
    def rotated(self, angle, degrees=False):
        if degrees:
            angle *= pi/180
        angle %= 2*pi
        
        #new_size = max(self.__arr.shape) * 3 // 2 # TODO

        new = np.zeros(self.__arr.shape, dtype=self.dtype)

        center = (new.shape[0] / 2, new.shape[1]/2)
        self_center = (self.__arr.shape[0] / 2, self.__arr.shape[1] / 2)

        def f(x, y):
            r = (x**2 + y**2) ** 0.5
            if r == 0: phi = 0
            elif y >= 0:
                phi = acos(x / r)
            else:
                phi = 2*pi - acos(x / r)
            self_i = self_center[0] + r * cos(phi - angle)
            self_j = self_center[1] + r * sin(phi - angle)
            return (self_i, self_j, 'fl')

        for i in trange(new.shape[0]):
            for j in range(new.shape[1]):
                x = i - center[0]
                y = j - center[1]
                new[i,j] = self[f(x, y)]
        
        return Timage(array=new)
    
    def distorted(self, ki, shape=None, scale=None):
        if shape is None and scale is None:
            shape = self.__arr.shape
        elif scale is not None:
            shape = (  int(self.__arr.shape[0]*scale), int(self.__arr.shape[1]*scale)  )
        new = np.zeros(shape, dtype=self.dtype)

        new_center = (shape[0]/2, shape[1]/2)

        for i in range(shape[0]):
            for j in range(shape[1]):
                new[(i, j)] = self[self.__distort(i, j, new_center, ki)]
        
        return Timage(array=new)

    

    def __distort(self, i, j, distorted_center, ki): # Brown-Conrady's even-order polynomial model

        r = (  (i-distorted_center[0])**2 + (j-distorted_center[1])**2  )**0.5
        
        k = sum(  ki[_] * r**(2*_) for _ in range(len(ki))  )
        if k < 0: return (float('inf'), float('inf'), 'fl')

        i_undistorted = self.__arr.shape[0] / 2 + (i-distorted_center[0]) * k # self.__arr.shape[0] / 2 == undistorted_center[0]
        j_undistorted = self.__arr.shape[1] / 2 + (j-distorted_center[1]) * k
        return (i_undistorted, j_undistorted, 'fl')
    