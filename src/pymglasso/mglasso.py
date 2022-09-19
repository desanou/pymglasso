"""
author: Edmond Sanou
Structure inspired from GGLasso Fabian Schaipp
"""

import numpy as np
from .utils import adjacency_matrix, beta_to_vector, beta_ols
from .utils import dist_beta
from .utils import merge_clusters
from .conesta import conesta
import sklearn.preprocessing

class mglasso_problem:
    """
    Class for Multiscale Graphical Lasso problem. The estimators can be accessed with ``self.solution``. See
    documentation of ``MGLassoEstimator`` for details.
        
    Parameters
    ----------
    X : 2d-array of shape (n,p)
        p-multivariate normal sample with n independent observations.
    
    lambda1 : float, positive
        Lasso regularization parameter.
    
    lambda2 : float, positive
        Fused-Group Lasso regularization parameter.
        
    lambda2_start : float, positive
        Starting value of lambda2
    
    lambda2_factor : float, positive
        Step used to update fused-group Lasso penalty.
    """

    ### INITIALIZAITON 
    def __init__(self,
                 X,
                 lambda1=0,
                 lambda2=0,
                 lambda2_start=0,
                 lambda2_factor=1.5):

        self.X = X.copy()
        self.lambda1 = lambda1
        self.lambda2_start = lambda2_start
        self.lambda2_factor = lambda2_factor

        self._check_xdata()

        #self.solution = MGLassoEstimator(X=self.X.copy(), n=self.n, p=self.p)
        self.solution = self.solve()

        return

    ### SANITY CHECKS
    def _check_xdata(self):
        assert len(self.X.shape) == 2, f"The data matrix has shape {self.X.shape}, MGLasso only handles 2d input"
        (self.n, self.p) = self.X.shape
        return

    ### SOLVER
    def solve(self, prec_=1e-2, type_="initial", solver='conesta', distance="euclidean", weights_=None,
              fuse_thresh=1e-3, maxit=1e3):
        """
        Solve the Multiscale Graphical Lasso problem.
        ##An instance of ``MGLassoEstimator`` is created and given to ``self.solution``.
        
        Parameters
        ----------
        prec_ : float, positive, optional
            Tolerance for the stopping criterion (duality gap).
        
        type_ : str, optional
            If "initial" use classical version without weights.
        
        solver : str
            Solver algorithm name.
            
        distance : str, optional
            Distance between regression vectors with permutation on symmetric coefficients.
        
        weights_ : 2d-array of shape (p,p)
            Weights matrix for total variation penalties.
        
        fuse_thresh : float, positive, optional
            Threshold for clusters fusion.
            
        maxit : int, optional
            Maximum number of iterations.
            
        Returns
        -------
        None.
        """

        assert solver in ["conesta"], "Only conesta solver is used. Future integrations are coming soon!"

        self.prec_ = prec_
        self.type_ = type_
        p = self.p
        self.weights_ = weights_
        self.distance = distance
        self.fuse_thresh = fuse_thresh
        self.maxit = maxit

        X = np.array(self.X.copy())
        X = sklearn.preprocessing.scale(X)
        clusters = np.arange(p)

        lambda2 = self.lambda2_start

        t = 0  # index for the out list.
        it = 0
        out = []
        clusters_prev = None
        ## End Initialisations

        ## Loop until all the variables merged
        while (len(np.unique(clusters)) > 1):
            clusters = np.arange(p)

            if (it == 0):
                beta_old = beta_to_vector(beta_ols(X))
                lambda2 = 0
            if (it == 1):
                lambda2 = self.lambda2_start

            Beta = conesta(X, self.lambda1, lambda2, beta_old, prec_=self.prec_, type_=self.type_, W_=self.weights_,
                           max_iter_=self.maxit)
            beta_old = beta_to_vector(Beta)

            ## Update distance matrix
            diffs = dist_beta(Beta, distance=self.distance)

            ## Clustering starts here
            pairs_to_merge = np.concatenate(np.where(diffs <= self.fuse_thresh)).reshape((2, -1)).T

            if (pairs_to_merge.shape[0] != 0):
                clusters = merge_clusters(pairs_to_merge, clusters)

            print("nclusters: ", len(np.unique(clusters)), "lambda2 :", lambda2)

            # if(clusters != clusters_prev):
            # out = np.append(out, {"Beta" : Beta, "clusters" : clusters.copy()})
            # names(out)[[t]] <- paste0("level", length(unique(clusters)))
            # t = t + 1

            out = np.append(out, {"Beta": Beta.copy(), "clusters": clusters.copy()})

            lambda2 = lambda2 * self.lambda2_factor
            it = it + 1

        #self.solution._set_solution(sol=out)

        return out

    def __repr__(self):
        return (
            " \n"
            + "MULTISCALE GRAPHICAL LASSO PROBLEM "
            + " \n"
            + "Penalty parameters:\n"
            + "lambda1: " + f"{self.lambda1}" + " \n"
            + "lambda2_start: " + f"{self.lambda2_start}" + " \n"
            + "lambda2_factor: " + f"{self.lambda2_factor}"
        )

# class MGLassoEstimator:
#     """
#     Estimator solution to Multiscale Graphical Lasso problem.
#
#     Parameters
#     ----------
#     X : 2d-array of shape (n,p)
#         p-multivariate normal sample with n independent observations.
#
#     n : int
#         number of observations.
#
#     p : int
#         Number of variables.
#     """
#
#     def __init__(self, X, n, p):
#         self.X = X.copy()
#         self.n = n
#         self.p = p
#
#         self.sol_ = None
#         self.sample_covariance_ = None
#
#         self.adjacency_ = None
#         return
#
#     def _set_solution(self, sol):
#         #self.sample_covariance_ = 1 / self.n * np.dot(self.X.transpose(), self.X)
#
#         self.sol_ = sol.copy()
#         #        self.calc_adjacency()
#
#         return

#    def calc_adjacency(self, t = 1e-10):
#        self.adjacency_ = adjacency_matrix(S = self.precision_, t = t)

#    def clustering_path(self, fuse_thresh):


#    def graph_path(self, lambda2_factor):
