import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Path(__file__) 获取的当前 py 文件的路径

import numpy as np

from Activation import sigmoid, identity_function

def init_network() -> dict[str: np.ndarray]:
    network: dict[str: np.ndarray] = dict()
    network['W1'] = np.array(object=[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array(object=[0.1, 0.2, 0.3])
    network['W2'] = np.array(object=[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array(object=[0.1, 0.2])
    network['W3'] = np.array(object=[[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array(object=[0.1, 0.2])
    return network

def forward(network: dict[str: np.ndarray], x: np.ndarray) -> np.ndarray:
    """x 必须为一维的长度为 2 的数组"""
    W1: np.ndarray = network['W1'] 
    W2: np.ndarray = network['W2'] 
    W3: np.ndarray = network['W3']
    b1: np.ndarray = network['b1']
    b2: np.ndarray = network['b2']
    b3: np.ndarray = network['b3']
    a1: np.ndarray = np.dot(a=x, b=W1) + b1
    z1: np.ndarray = sigmoid(x=a1)
    a2: np.ndarray = np.dot(a=z1, b=W2) + b2
    z2: np.ndarray = sigmoid(x=a2)
    a3: np.ndarray = np.dot(a=z2, b=W3) + b3
    y: np.ndarray = identity_function(x=a3)
    return y



if __name__ == "__main__":
    network = init_network()
    x: np.ndarray = np.array(object=[1.0, 0.5])
    y: np.ndarray = forward(network=network, x=x)
    print(y) # [0.31682708 0.69627909]