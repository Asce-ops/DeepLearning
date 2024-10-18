import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Path(__file__) 获取的当前 py 文件的路径
from typing import Callable
from collections import OrderedDict

import numpy as np

from Loss import cross_entropy_error
from Activation import softmax, sigmoid
from Gradient import numerical_gradient
import Layer

class TwoLayerNet: # 输入层、输出层和一个隐藏层
    # np.random.seed(314) # 设置随机种子
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float = 0.01) -> None:
        """初始化权重和偏置"""
        self.params: dict[str: np.ndarray] = dict()
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = weight_init_std * np.random.randn(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = weight_init_std * np.random.randn(output_size)
        self.train_loss_list: list[np.float64] = list() # 用于记录损失函数的下降过程

    def predict(self, X: np.ndarray) -> np.ndarray:
        a1: np.ndarray = np.dot(a=X, b=self.params["W1"]) + self.params["b1"]
        z1: np.ndarray = sigmoid(x=a1)
        a2: np.ndarray = np.dot(a=z1, b=self.params["W2"]) + self.params["b2"]
        output: np.ndarray = softmax(x=a2)
        return output
    
    def loss(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        pred: np.ndarray = self.predict(X=X)
        return cross_entropy_error(pred=pred, target=Y)
    
    def gradient(self, X: np.ndarray, Y: np.ndarray) -> dict[str: np.ndarray]:
        f: Callable = lambda W: self.loss(X=X, Y=Y)
        grads: dict[str: np.ndarray] = dict()
        grads["W1"] = numerical_gradient(func=f, x=self.params["W1"])
        grads["b1"] = numerical_gradient(func=f, x=self.params["b1"])
        grads["W2"] = numerical_gradient(func=f, x=self.params["W2"])
        grads["b2"] = numerical_gradient(func=f, x=self.params["b2"])
        return grads
    
    def train(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.01, step_num: int = 100, batch_size: int = 100) -> None:
        """随机梯度下降法"""
        for _ in range(step_num):
            batch_mask: np.ndarray[int] = np.random.choice(a=X.shape[0], size=batch_size) # 随机抽取小批量
            X_batch: np.ndarray = X[batch_mask]
            Y_batch: np.ndarray = Y[batch_mask]
            grads: dict[str: np.ndarray] = self.gradient(X=X_batch, Y=Y_batch)
            for param in grads:
                self.params[param] -= lr * grads[param]
            self.train_loss_list.append(self.loss(X=X_batch, Y=Y_batch)) # 记录学习过程

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        pred: np.ndarray = self.predict(X=X)
        Y_hat: np.ndarray = np.argmax(a=pred, axis=1) # 寻找每一行中最大值的位置
        Y_true: np.ndarray = np.argmax(a=Y, axis=1)
        return np.sum(a=(Y_hat==Y_true)) / X.shape[0]



class TwoLayerNet2(TwoLayerNet):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float = 0.01) -> None:
        super().__init__(input_size, hidden_size, output_size)
        """生成层"""
        self.layers: OrderedDict[str: Layer.HiddenLayer] = OrderedDict() # 有序字典
        self.layers["Affine1"] = Layer.Affine(W=self.params["W1"], b=self.params["b1"])
        self.layers["ReLU1"] = Layer.ReLU()
        self.layers["Affine2"] = Layer.Affine(W=self.params["W2"], b=self.params["b2"])
        self.lastLayer: Layer.SoftmaxWithLoss = Layer.SoftmaxWithLoss()

    def predict(self, X: np.ndarray) -> np.ndarray:
        layer: Layer.HiddenLayer
        for layer in self.layers.values():
            X = layer.forward(X=X)
        return X
    
    def loss(self, X: np.ndarray, Y: np.ndarray) -> np.float64:
        output: np.ndarray = self.predict(X=X)
        return self.lastLayer.forward(X=output, Y=Y) # 损失函数值被存储在 self.lastLayer 中
    
    def gradient(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """反向传播替代数值方法求梯度"""
        self.loss(X=X, Y=Y) # forward 更新损失函数值
        dout: np.ndarray = self.lastLayer.backward()
        layers: list[Layer.HiddenLayer] = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout=dout)
        grads: dict[str: np.ndarray] = dict()
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        return grads



if __name__ == "__main__":
    from time import time
    start: float = time()

    import matplotlib.pylab as plt
    
    from dataset.mnist import load_mnist
    
    (train_img, train_label), (test_img, test_label) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    # batch_mask: np.ndarray[int] = np.random.choice(a=train_img.shape[0], size=100)
    # train_img, train_label = train_img[batch_mask], train_label[batch_mask]
    
    net: TwoLayerNet2 = TwoLayerNet2(input_size=train_img.shape[1], hidden_size=50, output_size=train_label.shape[1])
    
    net.train(
        X=train_img, 
        Y=train_label, 
        # lr=0.01, 
        # step_num=1000
    )
    print("测试集上的准确率", net.accuracy(X=test_img, Y=test_label))
    end: float = time()
    print(f"耗时{end-start}秒")
    # for param in net.params:
    #     print(net.params[param])
    plt.plot(range(len(net.train_loss_list)), net.train_loss_list)
    plt.xlim(0, len(net.train_loss_list)) # 指定 x 轴的范围
    # plt.ylim(0, max(net.train_loss_list) + 1) # 指定 y 轴的范围
    plt.xlabel(xlabel="step_num")
    plt.ylabel(ylabel="loss value")
    plt.title(label="loss value with step_num")
    plt.show()
    
    grad_numerical = TwoLayerNet.gradient(self=net, X=train_img, Y=train_label)
    grad_backprop = net.gradient(X=train_img, Y=train_label)
    # 求各个权重的绝对误差的平均值
    for key in grad_numerical.keys():
        diff = np.average(a=np.abs(grad_backprop[key] - grad_numerical[key]) )
        print(key + ":" + str(diff))