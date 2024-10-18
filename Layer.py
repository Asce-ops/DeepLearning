from abc import ABC, abstractmethod
import numpy as np

from Activation import softmax
from Loss import cross_entropy_error

class HiddenLayer(ABC):
    """抽象隐藏层"""
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass

class ReLU(HiddenLayer):
    """ReLU 层"""
    def __init__(self) -> None:
        self.mask: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.mask = (X <= 0)
        output: np.ndarray = X.copy()
        output[self.mask] = 0
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        return dout

class Sigmoid(HiddenLayer):
    """Sigmoid 层"""
    def __init__(self) -> None:
        self.output: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx: np.ndarray = dout * self.output * (1 - self.output) # y = 1 / (1 + exp(-x)) 的导数为 y(1-y)
        return dx

class Affine(HiddenLayer):
    """仿射层"""
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W: np.ndarray = W
        self.b: np.ndarray = b
        self.X: np.ndarray = None
        self.dW: np.ndarray = None
        self.db: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        output: np.ndarray = np.dot(a=self.X, b=self.W) + self.b
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx: np.ndarray = np.dot(a=dout, b=self.W.T) # Y = XW + b 关于 X 的导数为 W.T
        self.dW = np.dot(a=self.X.T, b=dout)
        self.db = np.sum(a=dout, axis=0)
        return dx # 传递给前一层作为其 dout

class SoftmaxWithLoss:
    """cross_entropy_error(pred=softmax(x), target=y)"""
    def __init__(self) -> None:
        self.loss: np.float64 = None # 损失函数值
        self.output: np.ndarray = None # spftmax 层的输出
        self.Y: np.ndarray = None # 真实标签（独热编码）
    
    def forward(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        self.Y = Y
        self.output = softmax(x=X)
        self.loss = cross_entropy_error(pred=self.output, target=self.Y)
        return self.loss
    
    def backward(self) -> None: # 反向传播的起始层，无需接收 dout 参数
        if self.Y.ndim == 1:
            dx: np.ndarray = self.output - self.Y # cross_entropy_error(pred=softmax(x), target=y) 关于 x 的导数为 softmax(x) - y
        else:
            batch_size: int = self.Y.shape[0]
            dx: np.ndarray = (self.output - self.Y) / batch_size
        return dx # 传递给前一层作为其 dout
