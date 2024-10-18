import numpy as np

def AND(x1: 0 | 1, x2: 0 | 1) -> 0 | 1:
    """与门"""
    x: np.ndarray = np.array(object=[x1, x2])
    w: np.ndarray = np.array(object=[0.5, 0.5])
    b: float = -0.7
    tmp: np.float64 = np.sum(a=w*x) + b
    return 0 if tmp <= 0 else 1

def NAND(x1: 0 | 1, x2: 0 | 1) -> 0 | 1:
    """与非门"""
    return 1 - AND(x1=x1, x2=x2)

def OR(x1: 0 | 1, x2: 0 | 1) -> 0 | 1:
    """或门"""
    x: np.ndarray = np.array(object=[x1, x2])
    w: np.ndarray = np.array(object=[0.5, 0.5])
    b: float = -0.2
    tmp: np.float64 = np.sum(a=w*x) + b
    return 0 if tmp <= 0 else 1

def XOR(x1: 0 | 1, x2: 0 | 1) -> 0 | 1: # 异或门必须通过多层感知机来实现
    """异或门"""
    return AND(x1=NAND(x1=x1, x2=x2), x2=OR(x1=x1, x2=x2))

if __name__ == "__main__":
    print("与门")
    print(1, 1, AND(x1=1, x2=1))
    print(1, 0, AND(x1=1, x2=0))
    print(0, 1, AND(x1=0, x2=1))
    print(0, 0, AND(x1=0, x2=0))
    print("或门")
    print(1, 1, OR(x1=1, x2=1))
    print(1, 0, OR(x1=1, x2=0))
    print(0, 1, OR(x1=0, x2=1))
    print(0, 0, OR(x1=0, x2=0))
    print("与非门")
    print(1, 1, NAND(x1=1, x2=1))
    print(1, 0, NAND(x1=1, x2=0))
    print(0, 1, NAND(x1=0, x2=1))
    print(0, 0, NAND(x1=0, x2=0))
    print("异或门")
    print(1, 1, XOR(x1=1, x2=1))
    print(1, 0, XOR(x1=1, x2=0))
    print(0, 1, XOR(x1=0, x2=1))
    print(0, 0, XOR(x1=0, x2=0))