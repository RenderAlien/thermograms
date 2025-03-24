from typing import List

def median(arr: List[float]) -> float:
    '''Median of unsorted array'''
    '''time: O(n*log(n))'''
    srtd = sorted(arr)
    if len(arr)%2==1:
        return srtd[len(arr)//2]
    else:
        return sum(srtd[len(arr)//2-1:len(arr)//2+1])/2

def mean(arr: List[float]) -> float:
    '''Average value of array'''
    '''time: O(n)'''
    return sum(arr) / len(arr)

def variance(arr: List[float]) -> float:
    '''Дисперсия массива'''
    '''time: O(n)'''
    m = mean(arr)
    n = len(arr)

    return sum((m-arr[i])**2 for i in range(n)) / n

def standart_deviation(arr: List[float]) -> float:
    '''Среднеквадр отклонение'''
    '''time: O(n)'''
    return variance(arr) ** 0.5