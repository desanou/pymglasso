"""
Generate multivariate normal data from block diagonal covariance matrix.
"""
import random

import numpy as np
from matplotlib import pyplot as plt
import sklearn.preprocessing
from scipy.linalg import block_diag

def sample_block_diag_matrix(n = 20, K = 3, p = 6, rho = 0.5, seed = 2022):
    blocs = list()

    for j in range(0, K):
        bloc = np.full((p // K, p // K), rho)
        for i in range(0, p // K):
            bloc[i, i] = 1
        blocs.append(bloc)

    mat_covar = block_diag(*blocs)
    plt.imshow(mat_covar, interpolation='nearest')
    random.seed(seed)
    X = np.random.multivariate_normal(np.zeros(p), mat_covar, n)
    X = sklearn.preprocessing.scale(X)

    return mat_covar, X