from collections import OrderedDict
from typing import Callable
from math import ceil

import numpy as np

from Loss import cross_entropy_error
from Gradient import numerical_gradient
import Layer
import Optimizer

class MultiLayerNet:
    def __init__(self, input_size: int, hidden_size_list: list[int], output_size: int, activation: str = "ReLU", weight_init_std: float = 0.01) -> None:
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.hidden_size_list: list[int] = hidden_size_list
        self.hidden_layer_num: int = len(hidden_size_list)
        self.activation: str = activation
        self.weight_init_std: float = weight_init_std
        self.params: dict[str: np.ndarray] = dict()
        """初始化权重和偏置"""
        self.__init_weight()
        """生成层"""
        activation_layer: dict[str: Layer.Sigmoid | Layer.ReLU] = {"sigmoid": Layer.Sigmoid, "relu": Layer.ReLU}
        self.layers: OrderedDict[str: Layer.HiddenLayer] = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Layer.Affine(W=self.params["W" + str(idx)], b=self.params["b" + str(idx)])
            self.layers["Activation" + str(idx)] = activation_layer[self.activation]()
        idx: int = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Layer.Affine(W=self.params["W" + str(idx)], b=self.params["b" + str(idx)])
        self.last_layer: Layer.SoftmaxWithLoss = Layer.SoftmaxWithLoss()
        self.train_loss_list: list[np.float64] = list() # 用于记录损失函数的下降过程

    def __init_weight(self) -> None:
        all_size_list: list[int] = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            if self.activation == "ReLU":
                self.weight_init_std = np.sqrt(2.0 / all_size_list[idx - 1]) # 使用 ReLU 的情况下推荐的初始值
            elif self.activation == "Sigmoid":
                self.weight_init_std = np.sqrt(1.0 / all_size_list[idx - 1]) # 使用 Sigmoid 的情况下推荐的初始值
            self.params["W" + str(idx)] = self.weight_init_std * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params["b" + str(idx)] = np.zeros(shape=all_size_list[idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        layer: Layer.HiddenLayer
        for layer in self.layers.values():
            X = layer.forward(X=X)
        return X
    
    def loss(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        return cross_entropy_error(pred=self.predict(X=X), target=Y) # 损失函数值被存储在 self.lastLayer 中
    
    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        pred: np.ndarray = self.predict(X=X)
        Y_hat: np.ndarray = np.argmax(a=pred, axis=1) # 寻找每一行中最大值的位置
        if Y.ndim != 1:
            Y: np.ndarray = np.argmax(a=Y, axis=1)
        return np.sum(a=(Y_hat==Y)) / X.shape[0]
    
    def numerical_gradient(self, X: np.ndarray, Y: np.ndarray) -> dict[str: np.ndarray]:
        """数值方法求梯度, Y: 独热编码"""
        loss_W: Callable = lambda W: self.loss(X=X, Y=Y)
        grads: dict[str: np.ndarray] = dict()
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = numerical_gradient(func=loss_W, x=self.params["W" + str(idx)])
            grads["b" + str(idx)] = numerical_gradient(func=loss_W, x=self.params["b" + str(idx)])
        return grads
    
    def gradient(self, X: np.ndarray, Y: np.ndarray) -> dict[str: np.ndarray]:
        """反向传播求梯度, Y: 独热编码"""
        self.loss(X=X, Y=Y) # forward 更新损失函数值
        dout: np.ndarray = self.last_layer.backward()
        layers: list[Layer.HiddenLayer] = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout=dout)
        grads: dict[str: np.ndarray] = dict()
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = self.layers["Affine" + str(idx)].dW
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db
        return grads
    
    def train(self, X: np.ndarray, Y: np.ndarray, optimizer: Optimizer.Optimizer, step_num: int = 100, batch_size: int = 100) -> None:
        epoch: int = ceil(X.shape[0] / batch_size)
        for step in range(step_num):
            batch_mask: np.ndarray[int] = np.random.choice(a=X.shape[0], size=batch_size) # 随机抽取小批量
            X_batch: np.ndarray = X[batch_mask]
            Y_batch: np.ndarray = Y[batch_mask]
            grads: dict[str: np.ndarray] = self.gradient(X=X_batch, Y=Y_batch)
            optimizer.update(params=self.params, grads=grads)
            if step % epoch == 0:
                self.train_loss_list.append(self.loss(X=X_batch, Y=Y_batch)) # 记录学习过程



if __name__ == "__main__":
    from time import time
    start: float = time()

    import matplotlib.pylab as plt
    
    from dataset.mnist import load_mnist
    
    train_img: np.ndarray
    train_label: np.ndarray
    test_img: np.ndarray
    test_label: np.ndarray
    (train_img, train_label), (test_img, test_label) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    network: MultiLayerNet = MultiLayerNet(
                                            input_size=train_img.shape[1], 
                                            hidden_size_list=[100, 100, 100, 100], 
                                            output_size=10
                                            )
    optimizer: Optimizer.Optimizer = Optimizer.Adam()
    network.train(X=train_img, Y=train_label, optimizer=optimizer)
    print("测试集上的准确率", network.accuracy(X=test_img, Y=test_label))
    end: float = time()
    print(f"耗时{end - start}秒")
    plt.plot(range(len(network.train_loss_list)), network.train_loss_list)
    plt.xlim(0, len(network.train_loss_list)) # 指定 x 轴的范围
    plt.ylim(0, max(network.train_loss_list) + 1) # 指定 y 轴的范围
    plt.xlabel(xlabel="step_num")
    plt.ylabel(ylabel="loss value")
    plt.title(label="loss value with step_num")
    plt.show()