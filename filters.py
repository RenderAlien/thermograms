from tqdm import trange
from arr_manipulation import median

def median_filter(arr, wind_size):
    '''time: O(n*w*log(w))'''
    assert len(arr) >= wind_size
    n = len(arr)
    new = []
    
    for i in trange(n):
        if i < wind_size//2:
            new.append(median(arr[:wind_size//2 - i] + arr[:i + wind_size//2 - (wind_size+1)%2 + 1]))
        elif i >= n - wind_size//2 + (wind_size+1)%2:
            new.append(median(arr[i - wind_size//2 :] + arr[-(i - n + wind_size//2 - (wind_size+1)%2 + 1):]))
        else:
            new.append(median(arr[i - wind_size//2 : i + wind_size//2 - (wind_size+1)%2 + 1]))
    
    return new