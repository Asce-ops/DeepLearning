# 一个简单的神经网络（没有隐藏层）其损失函数的梯度下降

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Path(__file__) 获取的当前 py 文件的路径
from typing import Callable

import numpy as np

from Loss import cross_entropy_error
from Activation import softmax
from Gradient import gradient_descent

class SimpleNet:
    output_size: int = 3
    np.random.seed(314) # 设置随机种子
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        if X.ndim == 1:
            X = X.reshape((1, X.size))
            Y = Y.reshape((1, Y.size))
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y
        self.W: np.ndarray = np.random.randn(self.X.shape[1], SimpleNet.output_size) # 用高斯分布进行初始化
    
    def predict(self) -> np.ndarray:
        return np.dot(a=self.X, b=self.W)
    
    def loss(self) -> np.float64:
        output: np.ndarray = self.predict()
        pred: np.ndarray = softmax(x=output)
        loss: np.float64 = cross_entropy_error(pred=pred, target=self.Y)
        return loss

    def train(self) -> None:
        f: Callable = lambda W: self.loss()
        self.W = gradient_descent(func=f, init_x=net.W)


if __name__ == "__main__":
    x: np.ndarray = np.array([0.6, 0.9])
    t: np.ndarray = np.array([0, 0, 1]) # 正确解标签（one-hot 向量）
    net: SimpleNet = SimpleNet(X=x, Y=t)
    print(net.loss())
    net.train()
    print(net.W)
    print(softmax(x=np.dot(a=net.X, b=net.W)))