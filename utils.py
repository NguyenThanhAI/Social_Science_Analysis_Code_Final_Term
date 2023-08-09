
from typing import List, Tuple
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