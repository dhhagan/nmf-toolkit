

import numpy as np
from nonnegfac.nnls import nnlsm_blockpivot as nnlstsq

def censored_nnlstsq(A, B, M):
    """Solves nonnegative least-sqaures problem with missing data in B.

    Parameters
    ----------
    A : ndarray
        m x r matrix
    B : ndarray
        m x n matrix
    M : ndarray
        m x n binary matrix (zeros indicating missing values)
    
    Returns
    -------
    X : ndarray
        nonnegative r x n matrix that minimizes norm(M* (AX - B))
    """
    if A.ndim == 1:
        A = A[:, None]
    
    # n x r x 1 tensor
    rhs = np.dot(A.T, M*B).T[:, :, None]
    
    # n x r x r tensor
    T = np.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])
    
    X = np.empty((B.shape[1], A.shape[1]))
    
    for n in range(B.shape[1]):
        X[n] = nnlstsq(T[n], rhs[n], is_input_prod=True)[0].T
    
    return X.T


def cv_nmf(data, rank, M=None, p_holdout=0.1, tol=0.001, verbose=False, max_iter=50):
    """Perform cross-validation for NMF using a speckled holdout pattern.
    
    This code was adapted from Alex Williams at Standford (https://bit.ly/2KDqKoW).
    
    Parameters
    ----------
    data : ndarray
        m x n matrix (orginial data as a time-series)
    rank : int
        The desired output rank (i.e., the number of factors)
    M : ndarray, optional
        An m x n binary matrix where zeros indicate missing (i.e., held out) values, by default None
    p_holdout : float, optional
        All random data under this value will be set to False, by default 0.1
    tol : float, optional
        The tolerance for convergence - the solution has converged when the MSE of subsequent iterations
        are less than this number, by default 0.001
    verbose : bool, optional
        Debugging status, by default False
    max_iter : int, optional
        The maximum number of iterations for convergance, by default 50
    """
    # Create the masking matrix to decide which values to "hold out"
    if M is None:
        M = np.random.rand(*data.shape) > p_holdout

        # Check to make sure there are enough values in each row to avoid singular matrix issues
        if M.sum(axis=0).any() < rank:
            for i in range(M.shape[0]):
                if M[i].sum() < rank:
                    M[i, :] = [True]*M.shape[1]
        
    # Initialize U randomly
    U = np.random.rand(data.shape[0], rank)
    
    # Initialize the mean error
    mse = 100.
    converged = False
    
    # Fit the NMF
    for i in range(max_iter):
        Vt = censored_nnlstsq(U, data, M)
        U = censored_nnlstsq(Vt.T, data.T, M.T).T
        
        # Calculate the current iterations MSE
        tmp_mse = np.mean((np.dot(U, Vt) - data)**2)
        diff = mse - tmp_mse
        
        # Break out of the loop if converged
        if abs(diff) <= tol:
            converged = True
            break
        else:
            mse = tmp_mse
            
    if verbose:
        train_pct = 100 * M.sum() / M.size
        print (f"\tIter {i} = {diff:.5f}")
        print (f"\tTrain/Test: {train_pct:.2f}/{100-train_pct:.2f}")
        
    # Return the result and the train/test error
    resid = np.dot(U, Vt) - data
    train_err = np.mean(resid[M]**2)
    test_err = np.mean(resid[~M]**2)
    
    return U, Vt, train_err, test_err, converged