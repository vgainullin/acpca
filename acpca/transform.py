import numpy as np

from scipy.sparse.linalg import LinearOperator, eigsh


class CalAv(LinearOperator):
    """
    Compute eigenpairs from a specified region
    """
    def __init__(self, X, K, l, dtype="float32"):
        self.X = X
        self.K = K
        self.L = l
        self.dtype = np.dtype(dtype)
        self.shape = (X.shape[1],X.shape[1])
       
    def _matvec(self, v):
        A = np.identity(self.K.shape[0])-(self.L*self.K)
        B = self.X @ v.flatten(order="F")
        C = A @ B
        return self.X.T @ C
     
    def _rmatvec(self, v):
        A = np.identity(self.K.shape[0])-(self.L*self.K)
        B = self.X @ v.flatten(order="F")
        C = A @ B
        return self.X.T @ C

class ACPCA:
    def __init__(self, Y=None, n_components=2, L=0.0):
        self.n_components = n_components
        self.L = L
        self.Y = Y
        self.X = None
        self.v = None
        self.e = None
       
    def calkernel_(self, kernel="Linear"):
        """
        Take the cross-product of the transpose of a matrix.
        """
        # TODO implement gaussian kernel
        if kernel == "Linear":
            return self.Y @ self.Y.T
        else:
            raise NotImplementedError
   
    def calc_best_lambda(self, thresh=0.05):
        """
        Compute best lambda
        """
        K = self.calkernel_(self.Y)
        lambdas = np.arange(0, 10, 0.05)
        ratios = []
        for l in lambdas:
            cav = CalAv(self.X, K, l)
            res = eigsh(cav, k=2, which="LA")
            Xv = self.X @ res[1]
            A = Xv.T @ (self.K @ Xv)
            ratio  = np.diag(A) / np.diag(Xv.T @ Xv)
            ratios.append(ratio)
        ratios = np.vstack(ratios)
        tmp = np.where(np.apply_along_axis(np.sum, 1, ratios <= thresh) == ratios.shape[1])
        best_lambda = np.round(lambdas[np.min(tmp)], 4)
        return best_lambda
     
    def fit(self, X, y=None):
        """
        Compute the adjusted PCA of X given counfounding matrix Y
        """
        # TODO make sure data is centered and scaled
        # TODO implement scaling / centering
        self.X = X
        self.Y = self.make_confounder_matrix(self.Y)
        K = self.calkernel_()
        cav = CalAv(self.X, K, l=self.L)
        self.e, self.v = eigsh(cav, k=self.n_components, which="LA")
        return self

    def transform(self, X, y=None):
        return X @ self.v

    def make_confounder_matrix(self, labels):
        """
        Generate matrix of confounding factors for ac-pca
        """
        ncol = np.max(labels) + 1
        nrow = labels.shape[0]
        pos_val = 1 / ncol * (ncol - 1)
        neg_val = -(1 - pos_val)
        y = np.zeros((nrow, ncol), dtype=np.float32)
        for i, index in enumerate(labels):
            y[i][index] = pos_val    
        y[np.where(y== 0.0)] = neg_val
        return y
     
    def get_params(self, deep=True):
        return {
          'n_components': self.n_components,
          'Y': self.Y,
          'L': self.L,
          'v': self.v,
          'X': self.X,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self