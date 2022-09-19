import numpy as np


# minus and plus lines with reodering the coefficients
def minus_lines(i, j, Beta, ni=1, nj=1):
    Y_coeffs = Beta[i, :].copy()
    X_coeffs = Beta[j, :].copy()
    X_i = X_coeffs[i]
    X_coeffs[i] = X_coeffs[j]
    X_coeffs[j] = X_i
    return ni * Y_coeffs - nj * X_coeffs


def plus_lines(i, j, Beta, ni=1, nj=1):
    Y_coeffs = Beta[i, :].copy()
    X_coeffs = Beta[j, :].copy()
    X_i = X_coeffs[i]
    X_coeffs[i] = X_coeffs[j]
    X_coeffs[j] = X_i
    return ni * Y_coeffs + nj * X_coeffs


# distances Beta
def dist_beta(Beta, distance="euclidean"):
    K = Beta.shape[1]

    if (K != 1):
        diffs = np.ones((K, K)) * np.inf
        for i in range(K - 1):
            for j in range(i + 1, K):
                diffs[i, j] = np.linalg.norm(minus_lines(i, j, Beta))

        if (distance == "relative"):
            Dsum = np.ones((K, K))
            for i in range(K - 1):
                for j in range(i + 1, K):
                    Dsum[i, j] = np.linalg.norm(Beta[i, :]) + np.linalg.norm(Beta[j, :])

            diffs = diffs / Dsum

    else:
        diffs = np.zeros((1, 1))

    return diffs


# F Schaipp
def adjacency_matrix(S, t=1e-10):
    A = (np.abs(S) >= t).astype(int)
    # do not count diagonal entries as edges
    if len(S.shape) == 3:
        for k in np.arange(S.shape[0]):
            np.fill_diagonal(A[k, :, :], 0)
    else:
        np.fill_diagonal(A, 0)
    return A


def merge_clusters(pairs_to_merge, clusters):
    for l in range(pairs_to_merge.shape[0]):
        pair_to_merge = pairs_to_merge[l, :]

        i = np.min(pair_to_merge)
        j = np.max(pair_to_merge)

        if (i != j):
            # merge clusters
            clusters[clusters == j] = i
            clusters[clusters > j] = clusters[clusters > j] - 1

            # update the rest of the table with the new clusters
            pairs_to_merge[pairs_to_merge == j] = i
            pairs_to_merge[pairs_to_merge > j] = pairs_to_merge[pairs_to_merge > j] - 1

    return (clusters)


# vectorize beta matrix
# beta_to_vector <- function(beta_mat){
#   beta_mat <- as.matrix(beta_mat)
#   diag(beta_mat) <- NA
#   beta_mat <- as.vector(t(beta_mat))
#   beta_mat <- beta_mat[which(!is.na(beta_mat))]
#   beta_mat
# }

def beta_to_vector(beta_mat):
    beta_mat = np.array(beta_mat)
    beta_mat[np.diag_indices_from(beta_mat)] = np.nan
    beta_mat = beta_mat.flatten()
    beta_mat = beta_mat[~np.isnan(beta_mat)]

    return beta_mat


def beta_ols(X):
    """
    This function returns the OLS estimator matrix in a columnwise regression framework. 
    When X'X is singular then its pseudo-inverse is computed.
    """
    p = X.shape[1]
    beta = np.zeros((p, p))

    for i in range(0, p):
        W = np.delete(X, i, 1)
        IWW = np.linalg.pinv(np.dot(W.T, W))
        bi = np.linalg.multi_dot([IWW, W.T, X[:, i]])

        bi = np.insert(bi, i, 0)
        beta[i, :] = bi

    return beta