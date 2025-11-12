import numpy as np
from Timage import loadfile
from scipy.ndimage import map_coordinates
from tqdm import trange
from scipy.signal import convolve2d
from Timage import Timage



class Tseries:
    def __init__(self, *, path: str = None, array: np.ndarray = None):
        if array is not None:
            self.__arr = array.copy()
        elif path is not None:
            self.__arr = loadfile(path)
        else:
            raise ValueError('Path to .npy file or array must be provided to initialize Tseries')
        self.shape = self.__arr.shape
        self.dtype = self.__arr.dtype

    def save(self, path):
        np.save(path, self.__arr)
    
    @property
    def array(self):
        return self.__arr.copy()
    
    def __add__(self, other: "Tseries"):
        return self.__arr + other.array

    def __sub__(self, other: "Tseries"):
        return self.__arr - other.array
    
    def __radd__(self, other: "Tseries"):
        return other.array + self.__arr
    
    def __rsub__(self, other: "Tseries"):
        return other.array - self.__arr
    
    def __str__(self):
        return f'Series {self.__arr.shape}'
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return Timage(array=self.__arr[:, :, index])
    
    def heating_point(self, epsilon=0.1):
        avg = np.average(self.__arr, axis=(0,1))
        for i in range(len(avg)-1):
            if avg[i+1] - avg[i] > epsilon:
                return i
        
        raise RuntimeError('no such point exists')
    
    def maxima(self):
        '''Returns number of frame of maxima'''
        avg = np.average(self.__arr, axis=(0,1))
        return int(np.where(avg == np.max(avg))[0][0])
    
    def distorted(self, K, shape=None, scale=None):
        if shape is None and scale is None:
            shape = self.shape[:2]
        elif scale is not None:
            shape = (  int(self.shape[0]*scale), int(self.shape[1]*scale)  )
        
        distorted_center = np.array(shape, dtype=np.float64) / 2
        undistorted_center = np.array(self.shape[:2], dtype=np.float64) / 2
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

        distorted = np.zeros(shape=(*shape, self.shape[2]), dtype=self.dtype)

        for i in trange(self.shape[2]): # TODO

            flat_distorted = map_coordinates(
                self.__arr[:, :, i], 
                flat_interp_coordinates,
                order=1, #bilinear interpolation
                cval=0.0 # 0.0 is beyond self.__arr
            )
            distorted[..., i] = flat_distorted.reshape(shape).astype(self.dtype)

        return Tseries(array=distorted)

    def gaussian_blur(self, stddev=1, radius=3, circle = True):
        x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        G = np.exp(-(x**2 + y**2) / (2 * stddev**2)) / ((2 * np.pi)**0.5 * stddev)
        if circle: G[x**2 + y**2 > radius**2] = 0
        G /= np.sum(G) #kernel normalization

        new_arr = self.array

        for i in trange(new_arr.shape[2]):
            new_arr[:, :, i] = convolve2d(self.__arr[:, :, i], G, mode='same', boundary='symm').astype(self.dtype)
        
        return Tseries(array=new_arr)

    def std_defect_map(self, mean_method='avg') -> np.ndarray:
        '''This method normalizes and returns 2d array of stddevs: map[i, j] = sum( (norm[i,j,t] - avg(norm[i, j]))**2 )'''
        tser = Tseries(array=self.array[self.heating_point(): ])
        tau_m = tser.maxima()
        arr = tser.array

        norm = (arr - arr[:, :, 0].reshape(*arr.shape[:2], 1)) / (arr[:, :, tau_m].reshape(*arr.shape[:2], 1) - arr[:, :, 0].reshape(*arr.shape[:2], 1))
        if mean_method == 'avg':
            diff = norm - np.average(norm, axis=(0,1))
        elif mean_method == 'median':
            diff = norm - np.median(norm, axis=(0,1))
        else:
            raise ValueError(f"Incorrect mean_method: {mean_method}")
        diff **= 2
        diff = np.sum(diff, axis=2)
        return diff