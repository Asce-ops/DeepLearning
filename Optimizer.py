from abc import ABC, abstractmethod

import numpy as np

class Optimizer(ABC):
    """优化器接口"""
    @abstractmethod
    def update(self, params: dict[str: np.ndarray], grads: dict[str: np.ndarray]) -> None:
        """原地更新 params"""
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr: float = lr

    def update(self, params: dict[str: np.ndarray], grads: dict[str: np.ndarray]) -> None:
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr: float = lr
        self.momentum: float = momentum
        self.v: dict[str, np.ndarray] = None

    def update(self, params: dict[str: np.ndarray], grads: dict[str: np.ndarray]) -> None:
        if self.v is None:
            self.v = dict()
            for key, val in params.items():
                self.v[key] = np.zeros_like(a=val)
        for key in params:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.01, eplison: float = 1e-7) -> None:
        self.lr: float = lr
        self.h: dict[str: np.ndarray] = None
        self.eplison: float = eplison # 用于避免除零错误

    def update(self, params: dict[str: np.ndarray], grads: dict[str: np.ndarray]) -> None:
        if self.h is None:
            self.h = dict()
            for key, val in params.items():
                self.h[key] = np.zeros_like(a=val)
        for key in params:
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eplison)


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eplison: float = 1e-7) -> None:
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eplison: float = eplison # 用于避免除零错误
        self.iter: int = 0
        self.m: dict[str: np.ndarray] = None
        self.v: dict[str: np.ndarray] = None
        
    def update(self, params: dict[str: np.ndarray], grads: dict[str: np.ndarray]) -> None:
        if self.m is None:
            self.m: dict[str: np.ndarray] = dict()
            self.v: dict[str: np.ndarray] = dict()
            for key, val in params.items():
                self.m[key] = np.zeros_like(a=val)
                self.v[key] = np.zeros_like(a=val)
        self.iter += 1
        lr_t: float  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        for key in params:
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eplison)
