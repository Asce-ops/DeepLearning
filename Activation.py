from numbers import Number

import numpy as np

def sigmoid(x: Number | np.ndarray) -> np.float64 | np.ndarray:
    return 1 / (1 + np.exp(-x))

def relu(x: Number | np.ndarray) -> np.float64 | np.ndarray:
    return np.maximum(0, x)

def identity_function(x: Number | np.ndarray) -> Number | np.ndarray:
    return x

def softmax(x: np.ndarray) -> np.ndarray:
    c: np.float64 = np.max(a=x) # 避免出现上溢出
    exp_a: np.ndarray = np.exp(x - c) # exp_a 中的各个元素一定在 (0, 1] 之间
    sum_exp_a: np.float64 = np.sum(a=exp_a)
    y: np.ndarray = exp_a / sum_exp_a
    return y

if __name__ == "__main__":
    print(relu(x=np.array(object=[-1, 0 ,1])))
    print(softmax(x=np.array([0.3, 2.9, 4.0])))
    a: np.ndarray = np.array(object=[1010, 1000, 990])
    print(softmax(x=a))
    print(type(sigmoid(x=0.0)))