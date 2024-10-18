import numpy as np

def cross_entropy_error(pred: np.ndarray, target: np.ndarray) -> np.float64:
    "交叉熵误差"
    if pred.ndim == 1:
        pred = pred.reshape((1, pred.size))
        target = target.reshape((1, target.size))
    batch_size: int = pred.shape[0]
    eplison: float = 1e-7 # 避免出现 log 0 的情况
    return -np.sum(a=target * np.log(pred + eplison)) / batch_size

def mean_squared_error(pred: np.ndarray, target: np.ndarray) -> np.float64:
    "均方误差"
    if pred.ndim == 1:
        target = target.reshape((1, target.size))
        pred = pred.reshape((1, pred.size))
    batch_size: int = pred.shape[0]
    return 0.5 * np.sum(a=(target - pred)**2) / batch_size



if __name__ == "__main__":
    t: list[0|1] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y: list[float] = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(cross_entropy_error(pred=np.array(object=y), target=np.array(object=t)))
    print(mean_squared_error(pred=np.array(object=y), target=np.array(object=t)))