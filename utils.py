
from typing import List, Tuple, Optional
import numpy as np


def pca(x: np.ndarray, alpha: float=0.95) -> Tuple[np.ndarray, np.ndarray]:
    
    mu = np.mean(x, axis=0, keepdims=True)

    x_mu = x - mu

    cov_matrix = np.matmul(x_mu.T, x_mu) / x.shape[0]

    w, v = np.linalg.eig(cov_matrix)

    order = np.argsort(w)[::-1]

    w = w[order]
    v = v[:, order]

    rate = np.cumsum(w) / np.sum(w)

    r = np.where(rate >= alpha)

    U = v[:, :(r[0][0] + 1)]

    reduced_x = np.matmul(x, U)

    #print(reduced_x)

    return U, reduced_x


def construct_kernel(x: np.ndarray, type: str, sigma: Optional[float]=None, r: Optional[float]=None, gamma: Optional[float]=None, d: Optional[float]=None) -> np.ndarray:

    if type == "linear":
        return np.matmul(x, x.T)
    elif type == "gaussian_rbf":
        assert sigma is not None
        dist_matrix = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        square_dist_matrix = np.sum(dist_matrix**2, axis=2)
        return np.exp(-square_dist_matrix/(2*sigma**2))
    elif type == "polynomial":
        assert r is not None and gamma is not None and d is not None
        return (r + gamma * np.matmul(x, x.T))**d
    elif type == "sigmoid":
        assert r is not None and gamma is not None
        return np.tanh(r + gamma * np.matmul(x, x.T))
    else:
        raise ValueError("{} is not a supported kernel type".format(type))
    
    
def kernel_pca(x: np.ndarray, alpha: float=0.95, type: str="gaussian_rbf", sigma: Optional[float]=None, r: Optional[float]=None, gamma: Optional[float]=None, d: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    if type == "linear":
        K = construct_kernel(x=x, type=type)
    elif type == "gaussian_rbf":
        K = construct_kernel(x=x, type=type, sigma=sigma)
    elif type == "polynomial":
        K = construct_kernel(x=x, type=type, r=r, gamma=gamma, d=d)
    elif type == "sigmoid":
        K = construct_kernel(x=x, type=type, r=r, gamma=gamma)
    else:
        raise ValueError("{} is not a supported kernel type".format(type))
    #print(K)
    n = K.shape[0]

    assert K.shape[0] == K.shape[1]

    K = np.matmul(np.eye(n) - np.ones(shape=(n, n))/n, K)
    K = np.matmul(K, np.eye(n) - np.ones(shape=(n, n))/n)

    eta, c = np.linalg.eig(K)

    eta = np.real(eta)
    c = np.real(c)

    order = np.argsort(eta)[::-1]
    eta = eta[order]
    c = c[:, order]

    lamb_da = eta / n

    c = c / (np.sqrt(eta + 1e-8)[np.newaxis, :])

    rate = np.cumsum(lamb_da) / np.sum(lamb_da)

    #print(rate)

    r = np.where(rate >= alpha)

    C = c[:, :(r[0][0] + 1)]

    reduced_data = np.matmul(C.T, K).T

    #print(reduced_data.shape)
    #print(reduced_data)

    return C, reduced_data