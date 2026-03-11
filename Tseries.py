import numpy as np
from Timage import Timage, loadfile
from scipy.ndimage import map_coordinates, median_filter
from tqdm import trange
from scipy.signal import convolve2d
from typing import Tuple
import cv2
from os.path import splitext
from scipy.io import savemat
from sklearn.decomposition import PCA



class Tseries:
    def __init__(self, *, path: str = None, array: np.ndarray = None) -> None:
        if array is not None:
            self.__arr = array.copy()
        elif path is not None:
            self.__arr = loadfile(path)
        else:
            raise ValueError('Path to .npy file or array must be provided to initialize Tseries')
        self.shape = self.__arr.shape
        self.dtype = self.__arr.dtype
    
    @property
    def array(self) -> np.ndarray:
        return self.__arr.copy()
    
    def __add__(self, other: "Tseries") -> "Tseries":
        return Tseries(array=self.__arr+other.array)

    def __sub__(self, other: "Tseries") -> "Tseries":
        return Tseries(array=self.__arr-other.array)
    
    def __radd__(self, other: "Tseries") -> "Tseries":
        return Tseries(array=other.array+self.__arr)
    
    def __rsub__(self, other: "Tseries") -> "Tseries":
        return Tseries(array=other.array-self.__arr)
    
    def __mul__(self, other: "Tseries") -> "Tseries":
        return Tseries(array=self.__arr*other.array)
    
    def __str__(self) -> str:
        return f'Series {self.shape}'
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return Timage(array=self.__arr[:, :, index])
        else:
            return Tseries(array=self.__arr[index])
    
    def heating_point(self, epsilon: float = 0.1) -> int:
        """Find idx where the heating starts"""
        avg = np.average(self.__arr, axis=(0,1))
        for i in range(len(avg)-1):
            if avg[i+1] - avg[i] > epsilon:
                return i
        
        raise RuntimeError('no such point exists')
    
    def maxima(self) -> int:
        '''Returns number of frame of maxima'''
        avg = np.average(self.__arr, axis=(0,1))
        return int(np.where(avg == np.max(avg))[0][0])
    
    def distorted(self, K, shape=None, scale=None) -> "Tseries":
        """Distort thermograms via coefficients K"""
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

        for i in trange(self.shape[2]): # TODO but for some reason one map_coordinates(...) is slower than this loop

            flat_distorted = map_coordinates(
                self.__arr[:, :, i], 
                flat_interp_coordinates,
                order=1, #bilinear interpolation
                cval=0.0 # 0.0 is beyond self.__arr
            )
            distorted[..., i] = flat_distorted.reshape(shape).astype(self.dtype)

        return Tseries(array=distorted)

    def gaussian_blur(self, stddev=1, radius=3, circle = True) -> "Tseries":
        """Apply a gaussian filter"""
        x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        G = np.exp(-(x**2 + y**2) / (2 * stddev**2)) / ((2 * np.pi)**0.5 * stddev)
        if circle: G[x**2 + y**2 > radius**2] = 0
        G /= np.sum(G) #kernel normalization

        new_arr = self.array

        for i in trange(new_arr.shape[2]):
            new_arr[:, :, i] = convolve2d(self.__arr[:, :, i], G, mode='same', boundary='symm').astype(self.dtype)
        
        return Tseries(array=new_arr)
    
    def median_blur(self, radius=3) -> "Tseries":
        """Apply a median filter"""
        new_arr = np.zeros(shape=self.shape, dtype=self.dtype)
        
        for i in trange(new_arr.shape[2]):
            new_arr[:, :, i] = median_filter(self.__arr[:, :, i], size=(2*radius+1, 2*radius+1), mode='reflect')
        
        return Tseries(array=new_arr)
    
    def sharpness(self, radius=3, stddev=1) -> "Tseries":
        """Increase sharpness"""
        new = np.zeros(shape=self.__arr.shape, dtype=self.__arr.dtype)

        x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)) # gaussian kernel initialization
        G = np.exp(-(x**2 + y**2) / (2 * stddev**2)) / ((2 * np.pi)**0.5 * stddev)

        G /= np.sum(G) #kernel normalization

        G *= -1
        G[radius, radius] += 2

        for i in range(self.__arr.shape[2]):
            new[..., i] = convolve2d(self.__arr[..., i], G, mode='same', boundary='symm')

        return Tseries(array=np.clip(new, 0., 255.))

    def std_map(self, nd: str | Tuple[int] = 'avg', binarization: str | float = 'otsu') -> np.ndarray:
        """This method normalizes and returns 2d array of stddevs: map[i, j] = sum( (norm[i,j,tau] - nd of tau)**2 ) or its binarized version"""
        tser = Tseries(array=self.array[:, :, self.heating_point(): ])
        tau_m = tser.maxima()

        before_heating = self.__arr[:, :, 0:1] # before_heating = arr[:, :, 0].reshape(*arr.shape[:2], 1)
        
        peak = self.__arr[:, :, tau_m:tau_m+1] # peak = arr[:, :, tau_m].reshape(*arr.shape[:2], 1)

        norm = (self.__arr - before_heating) / (peak - before_heating)

        if nd == 'avg':
            diff = norm - np.average(norm, axis=(0,1))
        elif nd == 'median':
            diff = norm - np.median(norm, axis=(0,1))
        elif isinstance(nd, tuple) and len(nd) == 2:
            diff = norm - norm[nd[0], nd[1]] # unpacking is not available until 3.11
        else:
            raise ValueError(f"Incorrect nd: {nd}")
        
        diff **= 2
        diff = np.sum(diff, axis=2)

        if binarization == -1:
            return diff
        if binarization == 'otsu': 
            return self._otsu(diff)
        
        threshold = np.percentile(diff, 100-binarization)
        return (diff>=threshold).copy()

    def avg_time(self, frames=3) -> "Tseries":
        """Returns Tseries whose each frame is average along time axis of [frames] frames of the original Tseries"""
        if self.shape[2]//frames == 0:
            raise ValueError(f"frames = {frames} is too big for this series with time length = {self.shape[2]}")
        new = np.zeros(shape=(*self.shape[:2], self.shape[2]//frames), dtype=self.dtype)

        for i in trange(new.shape[2]):
            new[..., i] = np.average(self.__arr[..., i*frames:(i+1)*frames], axis=2)
        
        return Tseries(array=new)
    
    def homography_transform(self, src_points, dst_points, shape=None, scale=None):
        """Apply affine transformation"""
        # [[x'], [y'], [1]] = H * [[x], [y], [1]]
        if shape is None and scale is None:
            shape = self.shape[:2]
        elif scale is not None:
            shape = (  int(self.shape[0]*scale), int(self.shape[1]*scale)  )
        src_points, dst_points = np.array(src_points, dtype=np.float64), np.array(dst_points, dtype=np.float64)
        H = np.linalg.inv(cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)[0])
        
        # coordinate mesh
        i_arr, j_arr = np.mgrid[0:shape[0], 0:shape[1]]
        coordinates = np.array([i_arr.ravel(), j_arr.ravel(), np.ones(((shape[0]*shape[1])))])

        transformed_coords_with_w = H @ coordinates
        transformed_coords = np.array([transformed_coords_with_w[0] / transformed_coords_with_w[2], transformed_coords_with_w[1] / transformed_coords_with_w[2]])

        new = np.zeros(shape=(*shape, self.shape[2]), dtype=self.dtype)

        for i in range(self.shape[2]):

            flat_transformed = map_coordinates(
                self.__arr[:, :, i], 
                transformed_coords,
                order=1, #bilinear interpolation
                mode = 'constant',
                cval=0.0 # 0.0 is beyond self.__arr
            )
            new[:, :, i] = flat_transformed.reshape(shape).astype(self.dtype)

        return Tseries(array=new)

    def save(self, path):
        """Save Tseries"""
        filename, extension = splitext(path)
        if extension == '.npy':
            np.save(path, self.__arr)
        elif extension == '.mat':
            savemat(path, {'data': self.__arr})
        else:
            raise ValueError('Inappropriate file extension')
        
    def fft(self) -> np.ndarray:
        """Fast Fourier transformation"""
        return np.fft.fft(self.__arr, axis=2)

    def pca(self, n_components=4) -> np.ndarray:
        """Principal component analysis"""
        pca_model = PCA(n_components=n_components).fit(self.__arr.reshape(-1, self.shape[2]).T)
        return pca_model.components_.T.reshape(*self.shape[:2], -1)

    def _otsu(self, arr, bins = 1000):
        """Otsu binarization"""
        if len(arr.shape) != 2:
            raise ValueError(f"Array must be 2D matrix but not {len(arr.shape)}D")

        threshold = min(np.arange(arr.min(), arr.max(), (arr.max() - arr.min()) / bins), key=lambda x: self._otsu_check_threshold(arr, x))
        
        binarized = (arr>=threshold).copy()

        return binarized

    def _otsu_check_threshold(self, arr, threshold):
        return np.nansum([
            np.mean(cls) * np.var(arr, where=cls) for cls in [arr>=threshold, arr<threshold]
        ])