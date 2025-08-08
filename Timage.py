#from warnings import deprecated
#import random as rnd
#from typing import List

from PIL import Image
import numpy as np
from math import pi, acos, cos, sin
from tqdm import trange
from IPython.display import display
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, map_coordinates
from cv2 import findHomography
np_pi = np.float32(pi)


# coefficients for camera distortion correction
#CAM_K = [1, -0.0000015, 4.8*0.000001**2] 
CAM_K = np.array([1, 0, -0.0000013, 0, 3.3*0.000001**2], dtype=np.float64)



def merge(t1: "Timage", t2: "Timage", vertical=True):
    if vertical:
        return Timage(array=np.append(t1.array, t2.array, axis=0))
    else:
        return Timage(array=np.append(t1.array, t2.array, axis=1))



def __get_timage(n, file):
    arr = np.array([[file['A'][j][i][n] for j in range(len(file['A']))] for i in range(len(file['A']))], dtype=np.float32)
    return Timage(array=arr)
    


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
        self.shape = self.__arr.shape

        ###print('__init__ ' + str(time()-t))

    def __add__(self, other: "Timage") -> "Timage":
        if self.shape != other.shape:
            raise ValueError("Can't add images of different shapes.")
        out = self.__arr+other.__arr
        return Timage(array=out)
    
    def __repr__(self):
        display(self.__img)
        return ""
    
    def __getitem__(self, index):
        
        if isinstance(index, slice|int):
            return Timage(array=self.array[index])

        if len(index) == 3:
            x, y, _type = index
        elif len(index) == 2:
            if isinstance(index[0], slice) or isinstance(index[1], slice): 
                return Timage(array=self.array[index])
            x, y = index
            _type = None
        
        if _type is None and (  not isinstance(x, int|float) or not isinstance(y, int|float)  ):
            raise KeyError(
                f"Inappropriate type for coordinates of thermogram, {x, y, _type} cannot be parameters."
            )

        if _type == 'int' or isinstance(x, int) and isinstance(y, int):
            if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
                return self.__arr[x, y]
            else:
                return self.dtype(0)
        else:
            if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
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
        out = self.array
        rnd = np.random.random(size=self.shape)
        out[rnd <= intensity / 2] = 0
        out[(rnd > intensity / 2) & (rnd <= intensity)] = 255

        return Timage(array=out)
    
    def gaussian_noise(self, mean=0, stddev=32) -> "Timage":
        rnd =np.random.normal(mean, stddev, size = self.shape)
        out = self.__arr + np.minimum(255-self.__arr, rnd)
        return Timage(array=out)
        
    def defect_map(self, radius=4, stddev=1.5, contrast_level=0.997, direction=None, pallete=[[0,0,0],[255,255,255]]):

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

        #return 255 * directed / np.max(directed)

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
        ki = (self.shape[0] - 1) / (shape[0] - 1)
        kj = (self.shape[1] - 1) / (shape[1] - 1)

        for i in range(shape[0]):
            for j in range(shape[1]):
                new[i,j] = self[i*ki, j*kj]
        
        return Timage(array=new)
    
    def rotated(self, angle, degrees=False):
        if degrees:
            angle %= 360
            if angle % 90 == 0:
                if angle < 0: angle += 360
                if angle == 0: return Timage(array=self.array)
                elif angle == 90: return Timage(array=self.array.T[::-1])
                elif angle == 180: return Timage(array=self.array[::-1, ::-1])
                elif angle == 270: return Timage(array=self.array[::-1].T)
            angle *= pi/180
        else:
            angle %= 2*pi
        
        #new_size = max(self.__arr.shape) * 3 // 2 # TODO

        new = np.zeros(self.shape, dtype=self.dtype)

        center = (new.shape[0] / 2, new.shape[1]/2)
        self_center = (self.shape[0] / 2, self.shape[1] / 2)

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

        for i in range(new.shape[0]):
            for j in range(new.shape[1]):
                x = i - center[0]
                y = j - center[1]
                new[i,j] = self[f(x, y)]
        
        return Timage(array=new)
    
    #def distorted(self, K, shape=None, scale=None):
    #    if shape is None and scale is None:
    #        shape = self.shape
    #    elif scale is not None:
    #        shape = (  int(self.shape[0]*scale), int(self.shape[1]*scale)  )
    #    new = np.zeros(shape, dtype=self.dtype)
#
    #    new_center = np.array(shape, dtype=np.float64)/2
    #    undistorted_center = np.array(self.shape, dtype=np.float64)/2
    #    K = np.array(K, dtype=np.float64)
#
    #    for i in np.arange(shape[0], dtype=np.int16):
    #        for j in np.arange(shape[1], dtype=np.int16):
    #            new[i, j] = self[self.__distort(i, j, new_center, undistorted_center, K)]
    #    
    #    return Timage(array=new)
    
    def distorted(self, K, shape=None, scale=None):
        if shape is None and scale is None:
            shape = self.shape
        elif scale is not None:
            shape = (  int(self.shape[0]*scale), int(self.shape[1]*scale)  )

        distorted_center = np.array(shape, dtype=np.float64) / 2
        undistorted_center = np.array(self.shape, dtype=np.float64) / 2
        K = np.array(K, dtype=np.float64)

        # coordinate mesh
        i_arr, j_arr = np.mgrid[0:shape[0], 0:shape[1]]
        distorted_coordinates = np.stack([i_arr, j_arr], axis=-1)

        # vector to current pixel from center
        d = distorted_coordinates - distorted_center

        # distance to current pixel
        r = np.sqrt(np.sum(d**2, axis=-1))

        mult = np.polynomial.polynomial.polyval(r, K)

        interp_coordinates = undistorted_center + d * mult[..., np.newaxis]

        interp_i = interp_coordinates[..., 0]
        interp_j = interp_coordinates[..., 1]
        flat_interp_coordinates = np.array([interp_i.ravel(), interp_j.ravel()])

        flat_distorted = map_coordinates(
            self.__arr, 
            flat_interp_coordinates,
            order=1, #bilinear interpolation
            mode = 'constant',
            cval=0.0 # 0.0 is beyond self.__arr
        )
        distorted = flat_distorted.reshape(shape).astype(self.dtype)
        return Timage(array=distorted)
        
    def lin_transform(self, T, shape=None, scale=None):
        if shape is None and scale is None:
            shape = self.__arr.shape
        elif scale is not None:
            shape = (  int(self.__arr.shape[0]*scale), int(self.__arr.shape[1]*scale)  )
        new = np.zeros(shape, dtype=self.dtype)

        inv_T = np.linalg.inv(T)

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = j - shape[1] / 2
                y = -i + shape[0] / 2

                r = [[x],
                     [y]]
                
                undistorted_r = inv_T @ r
                undistorted_x, undistorted_y = undistorted_r[0][0], undistorted_r[1][0]
                undistorted_i = self.__arr.shape[0]/2 - undistorted_y
                undistorted_j = self.__arr.shape[0]/2 + undistorted_x
                new[i,j] = self[undistorted_i, undistorted_j, 'fl']
        
        return Timage(array=new)
    
    def homography_transform(self, src_points, dst_points, shape=None, scale=None):
        # [[x'], [y'], [1]] = H * [[x], [y], [1]]
        if shape is None and scale is None:
            shape = self.shape
        elif scale is not None:
            shape = (  int(self.shape[0]*scale), int(self.shape[1]*scale)  )
        src_points, dst_points = np.array(src_points, dtype=np.float64), np.array(dst_points, dtype=np.float64)
        H = np.linalg.inv(findHomography(srcPoints=src_points, dstPoints=dst_points)[0])
        
        # coordinate mesh
        i_arr, j_arr = np.mgrid[0:shape[0], 0:shape[1]]
        coordinates = np.array([i_arr.ravel(), j_arr.ravel(), np.ones(((shape[0]*shape[1])))])

        transformed_coords_with_w = H @ coordinates
        transformed_coords = np.array([transformed_coords_with_w[0] / transformed_coords_with_w[2], transformed_coords_with_w[1] / transformed_coords_with_w[2]])

        flat_transformed = map_coordinates(
            self.__arr, 
            transformed_coords,
            order=1, #bilinear interpolation
            mode = 'constant',
            cval=0.0 # 0.0 is beyond self.__arr
        )
        transformed = flat_transformed.reshape(shape).astype(self.dtype)
        return Timage(array=transformed)
        
                

    

    def __distort(self, i, j, distorted_center, undistorted_center, K): # Brown-Conrady's even-order polynomial model
        distorted_coords = [i, j]

        r = np.sqrt(np.sum(  (distorted_center - distorted_coords)**2  )) # distance from center
        
        mult = np.polynomial.polynomial.polyval(r, K) # K[0] + K[1] * r**1 + K[2] * r**2 + ... 
        if mult < 0: return (float('inf'), float('inf'), 'fl')

        undistorted_coords = undistorted_center + (distorted_coords - distorted_center) * mult

        return (float(undistorted_coords[0]), float(undistorted_coords[1]), 'fl')


#class Tseries:
#    def __init__(self, *, path=None, file=None, series=None):
#        if not file:
#            self.file = loadmat(path)
#        else:
#            self.file = file
#        self.series = defaultdict(lambda i: __get_timage(i, self.file))
#
#    def __getitem__(self, index):
#
#        if isinstance(index, slice):
#            return 