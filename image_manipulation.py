from tqdm import trange
from threading import Thread
import random as rnd
from math import exp, pi

from PIL import Image
import numpy as np

from arr_manipulation import *


class Timage:
    def __init__(self, image=None, np_arr=None):
        self.image = image
        self.arr = np_arr
    
    def __add__(self, summand):
        assert self.get_image().size == summand.get_image().size
        n, m = self.get_image().size
        self_arr = self.get_array()
        summand_arr = summand.get_array()
        new_arr = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                new_arr[i][j] = (self_arr[i][j] + summand_arr[i][j]) / 2
        
        return Timage(np_arr=new_arr)
    
    def get_image(self):
        if self.image is None:
            self.image = Image.fromarray(self.arr)
        return self.image
    
    def get_array(self):
        if self.arr is None:
            one_line = np.array(self.image.getdata())
            width, height = self.image.size
            self.arr = np.array([one_line[i * width:(i + 1) * width] for i in range(height)], dtype=np.uint8)
        return self.arr
    
    def median_filtered(self, wind_size=3):
        n, m = self.get_image().size
        new_arr = np.zeros((m,n), dtype=np.uint8)
        for i in trange(m):
            for j in range(n):
                if wind_size <= i < m - wind_size and wind_size <= j < n - wind_size:
                    neighs = []
                    for n_i in range(i - wind_size, i+wind_size+1):
                        neighs.extend(self.arr[n_i][j-wind_size:j+wind_size+1])
                    new_arr[i][j] = median(neighs)
                else:
                    new_arr[i][j] = self.arr[i][j]
        
        return Timage(np_arr=new_arr)
    
    def gaussian_filtered(self, blur=1, wind_size=3):
        self_arr = self.get_array()
        new_arr = self_arr.copy()

        G = {} #Gaussian kernel
        for m in range(-wind_size, wind_size+1):
            for n in range(-wind_size, wind_size+1):
                G[(m, n)] = exp(-(m**2 + n**2)/(2*blur**2)) / (2 * pi * blur**2)

        for i in trange(wind_size, len(new_arr)-wind_size):
            for j in range(wind_size, len(new_arr[0])-wind_size):
                new_arr[i][j] = sum(sum(G[(m, n)]*self_arr[i+m][j+n] for n in range(-wind_size, wind_size+1)) for m in range(-wind_size, wind_size+1))
        
        return Timage(np_arr=new_arr)
    
    def salt_and_pepper_noised(self, intensity=0.1):
        new_arr = self.get_array().copy()
        n, m = self.get_image().size
        
        for i in range(m):
            for j in range(n):
                r = rnd.random()
                if r <= intensity/2:
                    new_arr[i][j] = 0
                elif r <= intensity:
                    new_arr[i][j] = 255
        
        return Timage(np_arr=new_arr)

    def get_grayscale(self):
        self_arr = self.get_array()
        assert len(self_arr[0][0]) == 4
        n, m = self.get_image().size
        new_arr = np.zeros((m, n), dtype=np.uint8)

        for i in trange(m):
            for j in range(n):
                r, g, b, br = self_arr[i][j]
                new_arr[i][j] = (br/255)*(0.299*r + 0.587*g + 0.114*b)
        
        return Timage(np_arr=new_arr)