from PIL import Image
import numpy as np
from math import pi, acos, cos, sin
from tqdm import trange
from IPython.display import display
from scipy.io import loadmat, savemat
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, map_coordinates
import cv2
from os.path import splitext
import matplotlib.pyplot as plt
np_pi = np.float32(pi)


# coefficients for camera distortion correction
CAM_K = np.array([1, 0, -1e-07*(13245) /1000, 0, (32382)*1e-13 /1000], dtype=np.float64)


# .._PALETTE[256] не используется и может быть любым значением, но необходима для случая, когда self.__arr[i, j] == 255.0
WB_PALETTE = np.mgrid[:257, :3][0] 

IRON_PALETTE = np.array(
    [[0, 0, 0],[1, 0, 2],[2, 0, 5],[3, 0, 7],[5, 0, 10],[6, 0, 12],[7, 0, 15],[8, 0, 17],[10, 0, 20],[11, 0, 22],[12, 0, 25],[13, 0, 27],[15, 0, 30],[16, 0, 32],[17, 0, 35],[18, 0, 37],[20, 0, 40],[21, 0, 42],[22, 0, 45],[23, 0, 47],[25, 0, 50],[26, 0, 52],[27, 0, 55],[28, 0, 57],[30, 0, 60],[31, 0, 62],[32, 0, 65],[33, 0, 67],[35, 0, 70],[36, 0, 72],[37, 0, 75],[38, 0, 77],[40, 0, 80],[41, 0, 82],[42, 0, 85],[43, 0, 87],[45, 0, 90],[46, 0, 92],[47, 0, 95],[48, 0, 97],[50, 0, 100],[51, 0, 102],[52, 0, 105],[53, 0, 107],[55, 0, 110],[56, 0, 112],[57, 0, 115],[58, 0, 117],[60, 0, 120],[61, 0, 122],[62, 0, 125],[63, 0, 127],[65, 0, 126],[67, 0, 121],[69, 0, 115],[71, 0, 110],[73, 0, 104],[75, 0, 99],[77, 0, 93],[79, 0, 88],[81, 0, 82],[83, 0, 77],[85, 0, 71],[87, 0, 66],[89, 0, 60],[91, 0, 55],[93, 0, 49],[95, 0, 44],[97, 0, 38],[99, 0, 33],[101, 0, 27],[103, 0, 22],[105, 0, 16],[107, 0, 11],[109, 0, 5],[111, 0, 0],[113, 0, 0],[115, 0, 0],[117, 0, 0],[119, 0, 0],[121, 0, 0],[123, 0, 0],[125, 0, 0],[127, 0, 0],[130, 0, 0],[132, 0, 0],[134, 0, 0],[136, 0, 0],[138, 0, 0],[140, 0, 0],[142, 0, 0],[144, 0, 0],[146, 0, 0],[148, 0, 0],[150, 0, 0],[152, 0, 0],[154, 0, 0],[156, 0, 0],[158, 0, 0],[160, 0, 0],[162, 0, 0],[164, 0, 0],[166, 0, 0],[168, 0, 0],[170, 0, 0],[172, 0, 0],[174, 0, 0],[176, 0, 0],[178, 0, 0],[180, 0, 0],[182, 0, 0],[184, 0, 0],[186, 0, 0],[188, 0, 0],[190, 0, 0],[192, 0, 0],[194, 2, 0],[196, 5, 0],[198, 7, 0],[200, 10, 0],[202, 12, 0],[204, 15, 0],[206, 17, 0],[208, 20, 0],[210, 22, 0],[212, 25, 0],[214, 27, 0],[216, 30, 0],[218, 32, 0],[220, 35, 0],[222, 37, 0],[224, 40, 0],[226, 42, 0],[228, 45, 0],[230, 47, 0],[232, 50, 0],[234, 52, 0],[236, 55, 0],[238, 57, 0],[240, 60, 0],[242, 62, 0],[244, 65, 0],[246, 67, 0],[248, 70, 0],[250, 72, 0],[252, 75, 0],[254, 77, 0],[255, 80, 0],[255, 83, 0],[255, 86, 0],[255, 89, 0],[255, 92, 0],[255, 95, 0],[255, 98, 0],[255, 101, 0],[255, 104, 0],[255, 107, 0],[255, 110, 0],[255, 113, 0],[255, 116, 0],[255, 119, 0],[255, 122, 0],[255, 125, 0],[255, 128, 0],[255, 131, 0],[255, 134, 0],[255, 137, 0],[255, 140, 0],[255, 143, 0],[255, 146, 0],[255, 149, 0],[255, 152, 0],[255, 155, 0],[255, 158, 0],[255, 161, 0],[255, 164, 0],[255, 167, 0],[255, 170, 0],[255, 173, 0],[255, 176, 0],[255, 179, 0],[255, 182, 0],[255, 185, 0],[255, 188, 0],[255, 191, 0],[255, 194, 0],[255, 197, 0],[255, 200, 0],[255, 203, 0],[255, 206, 0],[255, 209, 0],[255, 212, 0],[255, 215, 0],[255, 218, 0],[255, 221, 0],[255, 224, 0],[255, 227, 0],[255, 230, 0],[255, 233, 0],[255, 236, 0],[255, 239, 0],[255, 242, 0],[255, 245, 0],[255, 248, 0],[255, 251, 0],[255, 254, 0],[255, 255, 5],[255, 255, 10],[255, 255, 15],[255, 255, 20],[255, 255, 25],[255, 255, 30],[255, 255, 35],[255, 255, 40],[255, 255, 45],[255, 255, 50],[255, 255, 55],[255, 255, 60],[255, 255, 65],[255, 255, 70],[255, 255, 75],[255, 255, 80],[255, 255, 85],[255, 255, 90],[255, 255, 95],[255, 255, 100],[255, 255, 105],[255, 255, 110],[255, 255, 115],[255, 255, 120],[255, 255, 125],[255, 255, 130],[255, 255, 135],[255, 255, 140],[255, 255, 145],[255, 255, 150],[255, 255, 155],[255, 255, 160],[255, 255, 165],[255, 255, 170],[255, 255, 175],[255, 255, 180],[255, 255, 185],[255, 255, 190],[255, 255, 195],[255, 255, 200],[255, 255, 205],[255, 255, 210],[255, 255, 215],[255, 255, 220],[255, 255, 225],[255, 255, 230],[255, 255, 235],[255, 255, 240],[255, 255, 245],[255, 255, 250],[255,255,255]]
    )

def read_ravi(path: str) -> np.ndarray:
    # https://github.com/sgascoin/readRavi/blob/master/readRavi.ipynb
    cap = cv2.VideoCapture(path) # Opens file

    # Fetch undecoded RAW video streams
    cap.set(cv2.CAP_PROP_FORMAT, -1)  # Format of the Mat objects. Set value -1 to fetch undecoded RAW video streams (as Mat 8UC1). [Using cap.set(CAP_PROP_CONVERT_RGB, 0) is not working
    
    cols  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frames width
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frames height
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get video frames count

    video = np.zeros((rows-1,cols,frames))

    for k in trange(0, frames):
        ret, frame = cap.read()  # Read next video frame (undecoded frame is read as long row vector).

        if not ret:
            break  # Stop reading frames when ret = False (after the last frame is read).

        # View frame as int16 elements, and reshape to cols x rows (each pixel is signed 16 bits)
        frame = frame.view(np.int16).reshape(rows, cols)

        video[:,:,k] = frame[1:, :]  # Ignore the first row.
    
    cap.release()
    
    return video.astype('float64')


def loadfile(path: str) -> dict | np.ndarray:
    """Load *.ravi, *.mat, *.npy files"""
    filename, extension = splitext(path)
    if extension == '.ravi':
        return (read_ravi(path) + 8192) / 32
    elif extension == '.mat':
        return loadmat(path)
    elif extension == '.npy':
        return np.load(path)
    else:
        raise ValueError('inappropriate file extension')


def merge(t1: "Timage", t2: "Timage", vertical=True) -> "Timage":
    # Merge two thermograms
    if vertical:
        return Timage(array=np.append(t1.array, t2.array, axis=0))
    else:
        return Timage(array=np.append(t1.array, t2.array, axis=1))


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
        out = self.array+other.array
        return Timage(array=out)
    
    def __sub__(self, other: "Timage") -> "Timage":
        new = self.array - other.array
        #if np.min(new) < 0: new -= np.min(new) # TODO подумать, стоит ли ставить условие или вычитать минимум всегда
        new[new>255] = 255
        return Timage(array=new)
    
    def __rsub__(self, other: "Timage") -> "Timage":
        return other.__sub__(self)
    
    def __truediv__(self, other: int | float) -> "Timage":
        return Timage(array=self.array / other)
    
    def __mul__(self, other: int | float) -> "Timage":
        return Timage(array=self.array * other)
    
    def __rmul__(self, other: int | float) -> "Timage":
        return self.__mul__(other)
    
    def __repr__(self) -> str:
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
            
    def __bilinear_interpolate(self, x, y) -> np.floating:
        # https://en.m.wikipedia.org/wiki/Bilinear_interpolation
        vx0 = self[int(x), int(y), 'int'] + (self[int(x)+1, int(y), 'int'] - self[int(x), int(y), 'int']) * (x % 1)
        vx1 = self[int(x), int(y)+1, 'int'] + (self[int(x)+1, int(y)+1, 'int'] - self[int(x), int(y)+1, 'int']) * (x % 1)
        value = vx0 + (vx1 - vx0) * (y % 1)
        return value
    
    def show(self, palette=WB_PALETTE, contrast_level: int = 0) -> Image.Image:
        """Show thermogram"""
        mean = np.mean(self.__arr[self.__arr > 0]) # [self.__arr > 0] is to avoid overlighting picture because of zeros
        f = lambda x: (x - mean) / (1 - contrast_level) + mean

        contrasted = f(self.__arr)
        contrasted[contrasted < 0] = 0
        contrasted[contrasted > 255] = 255
            
        # linear interpolation
        low_contrasted = np.floor(contrasted).astype('uint8')
        weights = contrasted - low_contrasted

        low_colored = np.take(palette, low_contrasted, axis=0) # low_colored[i,j] = palette[low_contrasted[i,j]]
        diff_colored = np.take(palette, low_contrasted+1, axis=0) - low_colored

        colored = low_colored + weights[..., None] * diff_colored

        
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
        """Apply a median filter"""
        # x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        out = median_filter(self.__arr, size=(2*radius+1, 2*radius+1), mode='reflect')
        
        return Timage(array=out)

    def gaussian_blur(self, stddev=1, radius=3, circle=True) -> "Timage":
        """Apply a gaussian filter"""
        x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)) # gaussian kernel initialization
        G = np.exp(-(x**2 + y**2) / (2 * stddev**2)) / ((2 * np.pi)**0.5 * stddev)
        if circle: G[x**2 + y**2 > radius**2] = 0
        G /= np.sum(G) #kernel normalization
        
        out = convolve2d(self.__arr, G, mode='same', boundary='symm').astype(self.dtype)
        return Timage(array=out)
    
    def sharpness(self, radius=3, stddev=1) -> "Timage":
        """Increase sharpness"""
        new = np.zeros(shape=self.__arr.shape, dtype=self.__arr.dtype)

        x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)) # gaussian kernel initialization
        G = np.exp(-(x**2 + y**2) / (2 * stddev**2)) / ((2 * np.pi)**0.5 * stddev)

        G /= np.sum(G) #kernel normalization

        G *= -1
        G[radius, radius] += 2

        new = convolve2d(self.__arr, G, mode='same', boundary='symm')

        return Timage(array=np.clip(new, 0., 255.))

    def salt_and_pepper_noise(self, intensity=0.1) -> "Timage":
        """Add noise consisting of 0 and 255"""
        out = self.array
        rnd = np.random.random(size=self.shape)
        out[rnd <= intensity / 2] = 0
        out[(rnd > intensity / 2) & (rnd <= intensity)] = 255

        return Timage(array=out)
    
    def gaussian_noise(self, mean=0, stddev=32) -> "Timage":
        """Add gaussian noise"""
        rnd =np.random.normal(mean, stddev, size = self.shape)
        out = self.__arr + np.minimum(255-self.__arr, rnd)
        return Timage(array=out)
        
    def detect_edges(self, radius=4, stddev=1.5, contrast_level=0.997, direction=None, palette=WB_PALETTE) -> Image.Image:
        """Detect edges with complex filters"""

        if not isinstance(palette, np.ndarray): palette = np.array(palette, dtype=np.float32)

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
        
        # linear interpolation
        low_contrasted = np.floor(contrasted).astype('uint8')
        weights = contrasted - low_contrasted

        low_colored = np.take(palette, low_contrasted, axis=0) # low_colored[i,j] = palette[low_contrasted[i,j]]
        diff_colored = np.take(palette, low_contrasted+1, axis=0) - low_colored

        colored = low_colored + weights[..., None] * diff_colored
        
        #Image.fromarray(colored.astype('uint8')).show()
        return Image.fromarray(colored.astype('uint8'))

    def resized(self, shape) -> "Timage":
        """Resize thermogram"""
        new = np.zeros(shape, dtype=self.dtype)
        ki = (self.shape[0] - 1) / (shape[0] - 1)
        kj = (self.shape[1] - 1) / (shape[1] - 1)

        for i in range(shape[0]):
            for j in range(shape[1]):
                new[i,j] = self[i*ki, j*kj]
        
        return Timage(array=new)
    
    def rotated(self, angle, degrees=False) -> "Timage":
        """Rotate thermogram"""
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
    
    def distorted(self, K, shape=None, scale=None) -> "Timage":
        """Distort thermogram via coefficients K"""
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
            cval=0.0 # 0.0 is beyond self.__arr
        )
        distorted = flat_distorted.reshape(shape).astype(self.dtype)
        return Timage(array=distorted)
        
    def lin_transform(self, T, shape=None, scale=None) -> "Timage":
        """Apply linear transformation"""
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
    
    def homography_transform(self, src_points, dst_points, shape=None, scale=None) -> "Timage":
        """Apply affine transformation"""
        # [[x'], [y'], [1]] = H * [[x], [y], [1]]
        if shape is None and scale is None:
            shape = self.shape
        elif scale is not None:
            shape = (  int(self.shape[0]*scale), int(self.shape[1]*scale)  )
        src_points, dst_points = np.array(src_points, dtype=np.float64), np.array(dst_points, dtype=np.float64)
        H = np.linalg.inv(cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)[0])
        
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
    
    def diapason_transform(self, in_diapason, out_diapason) -> "Timage":
        """Apply linear diapason transformation
        out diapason values must be in (0;1)"""
        in_start, in_end = in_diapason
        out_start, out_end = out_diapason
        res = np.zeros(shape=self.shape, dtype=self.dtype)

        left_mask = self.__arr<=in_start
        res[left_mask] = (self.__arr[left_mask] - self.__arr.min()) / (in_start - self.__arr.min()) * out_start

        mid_mask = (in_start<self.__arr)&(self.__arr<=in_end)
        res[mid_mask] = (self.__arr[mid_mask] - in_start) / (in_end - in_start) * (out_end - out_start) + out_start

        right_mask = in_end<self.__arr
        res[right_mask] = (self.__arr[right_mask] - in_end) / (self.__arr.max() - in_end) * (1 - out_end) + out_end
        return Timage(array=res)
    
    def save(self, path) -> None:
        """Save Timage"""
        filename, extension = splitext(path)
        if extension == '.npy':
            np.save(path, self.__arr)
        elif extension == '.mat':
            savemat(path, {'data': self.__arr})
        else:
            raise ValueError('Inappropriate file extension')
        
    def imshow(self, title=None, figsize=(12, 12), frameon=False, colorbar=False) -> None:
        plt.figure(figsize=figsize, frameon=frameon)
        plt.axis('off')
        if title: plt.title(title)
        plt.imshow(self.__arr)
        if colorbar: plt.colorbar()