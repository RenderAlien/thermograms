from tqdm import trange
import random as rnd
from math import exp, pi
from typing import List

from PIL import Image
import numpy as np

from IPython.display import display

class Timage:
    def __init__(self, *, image: Image = None, array: np.ndarray = None, dtype: np.dtype = None) -> None:
        """_summary_

        Args:
            image (_type_, optional): Image representing the array data. Defaults to None.
            array (np.ndarray, optional): 2D numpy array representing the image data. Defaults to None.

        Raises:
            ValueError: Exactly one of 'image' or 'arr' must be provided to initialize Timage.
        """
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
        if self.image.size != other.image.size:
            raise ValueError("Can't add images of different sizes.")
        n, m = self.image.size
        self_arr = self.array
        other_arr = other.array
        out_arr = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                out_arr[i][j] = (self_arr[i][j] + other_arr[i][j]) / 2

        return Timage(array=out_arr)
    
    def __repr__(self):
        display(self.__img)
        return ""

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
        n, m = self.__img.size
        new_arr = np.zeros((m, n), dtype=self.dtype)
        for i in trange(m, desc='Median blur: '):
            for j in range(n):
                neighs = []

                for n_i in range(i - radius, i + radius + 1):
                    if n_i >= m: row = m - n_i - 1
                    else: row = abs(n_i)
                    for n_j in range(j - radius, j + radius + 1):
                        if n_j >= n: col = n - n_j - 1
                        else: col = abs(n_j)

                        neighs.append(self.__arr[row][col])

                new_arr[i][j] = self.__flat_median(neighs)

        return Timage(array=new_arr)

    def gaussian_blur(self, blur=1, radius=3) -> "Timage":
        n, m = self.__img.size
        self_arr = [[float(el) for el in row] for row in self.array]
        new_arr = np.zeros((m, n), dtype=self.dtype)

        G = [[0]*(2*radius+1) for _ in range(2*radius+1)]  # Gaussian kernel
        for m in range(-radius, radius + 1):
            for n in range(-radius, radius + 1):
                G[m+radius][n+radius] = exp(-(m**2 + n**2) / (2 * blur**2)) / (2 * pi * blur**2)
        
        #kernel normalization
        div = sum(sum(row) for row in G)
        G = [[el/div for el in row] for row in G]

        for i in trange(len(new_arr), desc='Gaussian blur: '):
            for j in range(len(new_arr[0])):
                value = 0

                for m in range(-radius, radius + 1):
                    if i + m >= len(self_arr): row = len(self_arr) - (i + m) - 1
                    else: row = abs(i + m)
                    for n in range(-radius, radius + 1):
                        if j + n >= len(self_arr[0]): col = len(self_arr[0]) - (j + n) - 1
                        else: col = abs(j + n)

                        value += G[m+radius][n+radius] * self_arr[row][col]

                new_arr[i][j] = value

        return Timage(array=new_arr)

    def salt_and_pepper_noise(self, intensity=0.1) -> "Timage":
        new_arr = self.__arr.copy()
        n, m = self.__img.size

        for i in trange(m, desc='SaP noise: '):
            for j in range(n):
                r = rnd.random()
                if r <= intensity / 2:
                    new_arr[i][j] = 0
                elif r <= intensity:
                    new_arr[i][j] = 255

        return Timage(array=new_arr)
    
    def gaussian_noise(self, mean=0, stddev=32) -> "Timage":
        n, m = self.__img.size
        new_arr = np.zeros((m, n), dtype=np.uint8)
        self_arr = [[float(el) for el in row] for row in self.array]

        for i in trange(m, desc='Gaussian noise: '):
            for j in range(n):
                r = rnd.gauss(mu=mean, sigma=stddev)
                new_arr[i][j] = max(0, min(255, self_arr[i][j] + r))
        
        return Timage(array=new_arr)

    def __flat_median(self, arr: List[float]) -> float:
        '''Median of unsorted array'''
        '''time: O(n*log(n))'''
        srtd = sorted(arr)
        if len(arr)%2==1:
            return srtd[len(arr)//2]
        else:
            return sum(srtd[len(arr)//2-1:len(arr)//2+1])/2