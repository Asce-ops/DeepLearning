# 数值方法计算梯度

from numbers import Number
from typing import Callable

import numpy as np

def numerical_gradient(func: Callable, x: np.ndarray) -> np.ndarray:
    "数值梯度"
    delta: float = 1e-4
    shape: tuple[int] = x.shape
    x = x.flatten()
    grad: np.ndarray = np.zeros_like(a=x) # 生成和 x 形状相同的全零数组
    for idx in range(x.size): # x 是函数 func 的参数数组
        tmp: Number = x[idx]
        """func(x+delta) 的计算"""
        x[idx] = tmp + delta
        y1: Number = func(x.reshape(shape))
        """func(x-delta) 的计算"""
        x[idx] = tmp - delta
        y2: Number = func(x.reshape(shape))
        """偏导数的计算"""
        grad[idx] = (y1 - y2) / (2 * delta)
        x[idx] = tmp # 还原值
    return grad.reshape(shape)

def gradient_descent(func: Callable, init_x: np.ndarray, lr: float = 0.01, step_num: int = 100) -> np.ndarray:
    x: np.ndarray = init_x
    for _ in range(step_num):
        grad: np.ndarray = numerical_gradient(func=func, x=x)
        x -= lr * grad # x 和 init_x 引用的是同一个对象，更新 x 同时也会更新 init_x
    return x



if __name__ == "__main__":
    def function_2(x: np.ndarray):
        return x[0]**2 + x[1]**2
    
    init_x: np.ndarray = np.array(object=[-3.0, 4.0])
    minimum: np.ndarray = gradient_descent(func=function_2, init_x=init_x, lr=0.1, step_num=100)
    print(minimum)
    print(isinstance(function_2, Callable))