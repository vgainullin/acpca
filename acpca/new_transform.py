import numpy as np
from scipy.sparse.linalg import eigs

def acPCA(X, Y, lambda_, nPC, kernel, bandwidth, eval_, opts=None):
    """
    acPCA -- simultaneous dimension reduction and adjustment for confounding variation.

    Parameters:
    X : ndarray
        The n by p data matrix, where n is the number of samples, p is the number of variables.
        Missing values in X should be labeled as NaN. If a whole sample in X is missing, it should be removed.
    Y : ndarray
        The n by q confounder matrix, where n is the number of samples, q is the number of confounding factors.
        Missing values in Y should be labeled as NaN.
    lambda_ : float
        Tuning parameter, non-negative.
    nPC : int
        Number of principal components to compute.
    kernel : str
        The kernel to use, should be either 'linear' or 'gaussian'.
    bandwidth : float
        Bandwidth for gaussian kernel. Provide any number for 'linear' kernel, it won't affect the result.
    eval_ : bool
        Evaluate the significance of the PCs.
    opts : dict, optional
        Some other options:
        - centerX: center the columns in X. Default is True.
        - centerY: center the columns in Y. Default is True.
        - scaleX: scale the columns in X to unit standard deviation. Default is False.
        - scaleY: scale the columns in Y to unit standard deviation. Default is False.
        - numPerm: the number of permutations to evaluate the significance of the PCs. Default is 100.
        - alpha: the significance level. Default is 0.05.

    Returns:
    obj : dict
        Contains the results of the acPCA computation.
    """
    # Set default options if not provided
    if opts is None:
        opts = {
            'centerX': True,
            'centerY': True,
            'scaleX': False,
            'scaleY': False,
            'numPerm': 100,
            'alpha': 0.05
        }

    # Check for missing samples in X
    Xmis = np.sum(~np.isnan(X), axis=1)
    if np.any(Xmis == 0):
        raise ValueError('Some samples in X are missing')

    nX, p = X.shape
    nY, _ = Y.shape

    # Check if the number of samples in X and Y match
    if nX != nY:
        raise ValueError('The numbers of samples in X and Y do not match')

    # Check if lambda is non-negative
    if lambda_ < 0:
        raise ValueError('lambda should be non-negative')

    # Center the X matrix
    if opts['centerX']:
        X = X - np.nanmean(X, axis=0)

    # Center the Y matrix
    if opts['centerY']:
        Y = Y - np.nanmean(Y, axis=0)

    # Scale the X matrix
    if opts['scaleX']:
        Xsd = np.nanstd(X, axis=0)
        Xsd[Xsd == 0] = 1
        X = X / Xsd

    # Scale the Y matrix
    if opts['scaleY']:
        Ysd = np.nanstd(Y, axis=0)
        Ysd[Ysd == 0] = 1
        Y = Y / Ysd

    # Input the missing values in X and Y with the mean
    X[np.isnan(X)] = np.nanmean(X)
    Y[np.isnan(Y)] = np.nanmean(Y)

    # Calculate the kernel matrix
    K = calkernel(Y, kernel, bandwidth)

    # Define the function to calculate the adjusted covariance matrix
    def calAv(v):
        return (X.T - lambda_ * X.T @ K) @ (X @ v)

    # Compute the eigenvectors and eigenvalues
    V, D = eigs(calAv, k=nPC, which='LA')

    # Extract the eigenvalues
    eigenX = np.diag(D).real

    # Project the data
    Xv = X @ V

    # Calculate variance in the projected data
    varX = np.var(Xv, axis=0)

    # Calculate total variance in X
    totvar = np.sum(np.var(X, axis=0))

    # Calculate the percentage of variance explained
    varX_perc = varX / totvar

    # Initialize permutation results
    eigenXperm = np.nan
    varXperm = np.nan
    varXperm_perc = np.nan
    sigPC = np.nan

    # Evaluate the significance of PCs if required
    if eval_:
        eigenXperm = np.zeros((opts['numPerm'], nPC))
        varXperm = np.zeros((opts['numPerm'], nPC))
        varXperm_perc = np.zeros((opts['numPerm'], nPC))

        for i in range(opts['numPerm']):
            Xperm = X[np.random.permutation(nX), :]
            def calAvperm(v):
                return (Xperm.T - lambda_ * Xperm.T @ K) @ (Xperm @ v)
            Vperm, Dperm = eigs(calAvperm, k=nPC, which='LA')
            eigenXperm[i, :] = np.diag(Dperm).real
            Xpermv = Xperm @ Vperm
            varXperm[i, :] = np.var(Xpermv, axis=0)
            varXperm_perc[i, :] = varXperm[i, :] / totvar

        labs1 = np.where(eigenX < np.quantile(eigenXperm, 1 - opts['alpha'], axis=0))[0]
        sigPC1 = 0 if len(labs1) == 0 else np.min(labs1) - 1

        labs2 = np.where(varX < np.quantile(varXperm, 1 - opts['alpha'], axis=0))[0]
        sigPC2 = 0 if len(labs2) == 0 else np.min(labs2) - 1

        sigPC = max(sigPC1, sigPC2)

    # Prepare the output object
    obj = {
        'Xv': Xv,
        'v': V,
        'eigenX': eigenX,
        'varX': varX,
        'varX_perc': varX_perc,
        'eigenXperm': eigenXperm,
        'varXperm': varXperm,
        'varXperm_perc': varXperm_perc,
        'sigPC': sigPC,
        'lambda': lambda_,
        'kernel': kernel
    }

    if kernel == 'gaussian':
        obj['bandwidth'] = bandwidth

    return obj

def calkernel(Y, kernel, bandwidth):
    """
    Calculate the kernel matrix.

    Parameters:
    Y : ndarray
        The n by q confounder matrix.
    kernel : str
        The kernel to use, should be either 'linear' or 'gaussian'.
    bandwidth : float
        Bandwidth for gaussian kernel. Not used for linear kernel.

    Returns:
    K : ndarray
        The kernel matrix.
    """
    if kernel == 'linear':
        # Linear kernel: simply the dot product of Y with its transpose
        K = Y @ Y.T
    elif kernel == 'gaussian':
        # Gaussian kernel: compute the pairwise squared Euclidean distances
        sq_dists = np.sum(Y**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * (Y @ Y.T)
        # Apply the Gaussian function
        K = np.exp(-sq_dists / (2 * bandwidth**2))
    else:
        raise ValueError("Unsupported kernel type. Use 'linear' or 'gaussian'.")

    return K