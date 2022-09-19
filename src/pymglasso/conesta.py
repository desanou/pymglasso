"""
author: Edmond Sanou
"""

from scipy import sparse
import numpy as np
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import sklearn.preprocessing 

def Ak_from_pairs(k,p,type_,W):
    Ak = sparse.lil_matrix((int(p*(p-1)/2),p*p))
    ij=0
    if W is None:
        W = np.ones((p,p))
    if type_=="initial" or type_=="pcor" or type_=="adapt":
        for i in range(0,p-1):
            for j in range(i+1,p):
                #print(i*p+k,j*p+k,ij)
                if (i==k)|(j==k):
                    Ak[ij,i*p+j]=1*W[i,j]
                    Ak[ij,j*p+i]=-1*W[i,j]
                else:
                    Ak[ij,i*p+k]=1*W[i,j]
                    Ak[ij,j*p+k]=-1*W[i,j]
                ij=ij+1

    elif type_=="wr":# discarding beta_ij and beta_ji
        for i in range(0,p-1):
            for j in range(i+1,p):
                if (i==k)|(j==k):
                    Ak[ij,i*p+j]=0
                    Ak[ij,j*p+i]=0
                else:
                    Ak[ij,i*p+k]=1
                    Ak[ij,j*p+k]=-1
                ij=ij+1

    to_keep = list(set(range(Ak.shape[1]))-set(range(0,p*p,p+1)))
    Aknew = sparse.lil_matrix(sparse.csr_matrix(Ak)[:,to_keep])
    return(Aknew)

def linear_operator_from_num_variables(num_variables, type_, W):
    """Generates the linear operator for the TV lasso Nesterov function
    from number of variables.

    Parameters:
    ----------
    num_variables : Integer. The total number of variables, including the
            intercept variable(s).

    """
    A = list()
    for k in range(0,num_variables):
        Ak = Ak_from_pairs(k,num_variables,type_,W)
        A.append(Ak.tocsr())
    return A

def beta2Beta(beta,p): 
    Beta=np.zeros((p,p))
    for j in range(0,(p-1)):
        for i in range(0,p):
            k=i
            l=j
            if j>=i:
                l=j+1
            Beta[k,l]=beta[i*(p-1)+j]
    return(Beta)  

def precision2regression(K):
    p=K.shape[0]
    M=np.zeros((p,p))
    for i in range(0,p):
        for j in range(0,p):
            if i!=j:
                M[i,j]= - K[i,j]/K[i,i]
    return(M)     

def conesta(X, lambda1, lambda2, beta_warm, type_, W_, mean_ = False, max_iter_=1e4, prec_=1e-2):
    # """
    # This is a Continuation with Nesterov smoothing in a Shrinkage-Thresholding Algorithm, (Hadj-Selem et al. 2019)
    # for the Multiscale Graphical Lasso problem.
    #
    # .. math::
    #    \hat{\boldsymbol{\beta}} \in \underset{\boldsymbol{\beta}}{\operatorname{argmin}} \frac{1}{2} \sum_{i=1}^p \left \lVert \mathbf{X}^i - \mathbf{X}^{\setminus i} \boldsymbol{\beta}^i \right \rVert_2 ^2  + \lambda_1 \sum_{i = 1}^p  \left \lVert \boldsymbol{\beta}^i \right \rVert_1 + \lambda_2 \sum_{i < j} \left \lVert \boldsymbol{\beta}^i - \tau_{ij}(\boldsymbol{\beta}^j) \right \rVert_2
    #
    # Parameters
    # ----------
    # X : 2d-array of shape (n,p)
    #     p-multivariate normal sample with n independent observations.
    #
    # lambda1 : float, positive
    #     Lasso regularization parameter.
    #
    # lambda2 : float, positive
    #     Fused-Group Lasso regularization parameter.
    #
    # beta_warm : 1d-array of shape (1,p*(p-1))
    #     Warm initialization vector.
    #
    # type_ : str
    #     By default set to initial version which doesn't use weights.
    #
    # W_ : 2d-array of shape (p,p)
    #     Weights matrix for total variation terms.
    #
    # mean_ : boolean, optional
    #     Whether  the optimization function is weighted by the inverse of sample size. The default is False.
    #
    # max_iter_ : int, optional
    #     Maximum number of iterations. The default is 1000.
    #
    # prec_ : float, positive
    #     Precision.
    #
    # Returns
    # -------
    # Beta : 2d-array of shape (p,p).
    #     Matrix of regression vectors. Line k of the matrix represents the coefficients obtained from the L1-L2 penalized regression of variable k on the others.
    # """
    #
    X=np.array(X)
    n=X.shape[0]
    p=X.shape[1]
  
    X=sklearn.preprocessing.scale(X)
    y=X.reshape(n*p,1,order='F')
    Xvec=np.delete(np.kron(np.identity(p),X),range(0,p*p,p+1),axis=1)
    A_=linear_operator_from_num_variables(p, type_, W_)
  
    if beta_warm is not None:
        beta_warm = np.array(beta_warm)
        hgmm = estimators.LinearRegressionL1L2TV(l1 = lambda1, l2 = 0.0, tv = lambda2, A = A_, start_vector=beta_warm,
                                           algorithm=algorithms.proximal.CONESTA(max_iter=max_iter_, eps=prec_), mean=mean_)
    if beta_warm is None:
        hgmm = estimators.LinearRegressionL1L2TV(l1 = lambda1, l2 = 0.0, tv = lambda2, A = A_,
                                           algorithm=algorithms.proximal.CONESTA(max_iter=max_iter_, eps=prec_), mean=mean_)
    
    res = hgmm.fit(Xvec,y)
    Beta=beta2Beta(res.beta,p)
    
    return(Beta)
