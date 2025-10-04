import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class CalAv(LinearOperator):
    """
    Compute eigenpairs from a specified region
    """
    def __init__(self, X, K, l, dtype="float32", kernel="linear", bandwidth=None, degree=3, coef0=1, gamma=0.01):
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

class ACPCA(BaseEstimator, TransformerMixin):
    def __init__(self, Y=None, n_components=2, L=0.0, lambda_method='original',
                 preprocess=True, center_x=True, scale_x=False, 
                 center_y=False, scale_y=False,
                 kernel="linear", bandwidth=None, gamma=0.01, coef0=1, degree=3,
                 use_implicit=True):
        """
        Parameters:
        -----------
        Y : array-like, optional
            Confounding labels
        n_components : int
            Number of components to keep
        L : float
            Lambda value. If -1, best lambda will be calculated
        lambda_method : str
            Method to calculate best lambda: 'original', 'silhouette'
        preprocess : bool
            Whether to apply preprocessing (imputation and scaling)
        center_x, scale_x : bool
            Whether to center/scale X matrix (only if preprocess=True)
        center_y, scale_y : bool
            Whether to center/scale confounding matrix (only if preprocess=True)
        kernel : str
            Kernel type: "linear" or "gaussian"
        bandwidth : float, optional
            Bandwidth for gaussian kernel
        use_implicit : bool, default=True
            Whether to use implicit formation of adjusted covariance matrix
            for better memory efficiency with large datasets
        """
        self.n_components = n_components
        self.L = L
        self.Y = Y
        self.lambda_method = lambda_method
        self.preprocess = preprocess
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.use_implicit = use_implicit
        
    def calkernel_(self):
        """
        Calculate kernel matrix for confounding factors.
        
        Parameters:
        -----------
        kernel : str
            Kernel type: "linear", "gaussian", "polynomial", or "sigmoid"
        
        Returns:
        --------
        K : array-like
            Kernel matrix
        """
        if self.kernel == "linear":
            return self.confounding_matrix @ self.confounding_matrix.T
        
        elif self.kernel == "gaussian":
            if self.bandwidth is None:
                # Median heuristic for bandwidth selection if not specified
                pairwise_dists = np.sum((self.confounding_matrix[:, None, :] - 
                                       self.confounding_matrix[None, :, :]) ** 2, axis=-1)
                self.bandwidth = np.median(pairwise_dists[pairwise_dists > 0] ** 0.5)
            
            # Calculate pairwise squared Euclidean distances
            squared_dists = np.sum((self.confounding_matrix[:, None, :] - 
                                    self.confounding_matrix[None, :, :]) ** 2, axis=-1)
            
            # Apply Gaussian kernel
            K = np.exp(-squared_dists / (2 * self.bandwidth ** 2))
            return K
        
        elif self.kernel == "polynomial":
            # Default parameters for polynomial kernel
            degree = getattr(self, 'degree', 3)  # polynomial degree
            coef0 = getattr(self, 'coef0', 1)    # bias term
            
            # Calculate polynomial kernel: (x·y + coef0)^degree
            linear_kernel = self.confounding_matrix @ self.confounding_matrix.T
            K = (linear_kernel + coef0) ** degree
            return K
        
        elif self.kernel == "sigmoid":
            # Default parameters for sigmoid kernel
            gamma = getattr(self, 'gamma', 0.01)  # scale parameter
            coef0 = getattr(self, 'coef0', 1)     # bias term
            
            # Calculate sigmoid kernel: tanh(gamma * x·y + coef0)
            linear_kernel = self.confounding_matrix @ self.confounding_matrix.T
            K = np.tanh(gamma * linear_kernel + coef0)
            return K
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}. Choose 'linear', 'gaussian', 'polynomial', or 'sigmoid'")

    def _compute_adjusted_covariance_implicit(self, X, K, lambda_):
        """
        Compute adjusted covariance matrix implicitly using matrix-vector products.
        Returns a LinearOperator that represents the adjusted covariance matrix.
        """
        def matvec(v):
            # Compute (X.T - lambda * X.T @ K) @ (X @ v) efficiently
            Xv = X @ v.reshape(-1, 1)  # Ensure v is a column vector
            KXv = K @ Xv
            return (X.T @ Xv - lambda_ * X.T @ KXv).flatten()  # Return flattened array

        def rmatvec(v):
            # For symmetric matrices, rmatvec is the same as matvec
            return matvec(v)

        n_features = X.shape[1]
        return LinearOperator(
            shape=(n_features, n_features),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=X.dtype
        )
   
    def calc_best_lambda(self, thresh=0.05):
        """
        Compute best lambda using the original method from the AC-PCA paper.
        
        This method finds the smallest lambda value that sufficiently removes
        the confounding variation by analyzing the ratio between:
        1) The projection of transformed data onto the confounding space
        2) The magnitude of the transformed data
        
        Parameters:
        -----------
        thresh : float, default=0.05
            Threshold for the ratio. Lower values mean stronger batch effect removal.
            
        Returns:
        --------
        float
            Best lambda value that removes confounding effects
        
        Notes:
        ------
        The method stores optimization results in self.lambda_scores_ containing:
        - lambdas: array of tested lambda values
        - ratios: corresponding ratio values for each lambda
        - method: identifier for the optimization method used
        """
        K = self.calkernel_()
        lambdas = np.arange(0, 10, 0.05)
        ratios = []
        
        for l in lambdas:
            # Always use implicit formation for consistency when use_implicit=True
            if self.use_implicit:
                adj_cov_op = self._compute_adjusted_covariance_implicit(self.X, K, l)
            else:
                adj_cov_op = CalAv(self.X, K, l)
            
            # Compute eigendecomposition
            eigenvalues, eigenvectors = eigsh(adj_cov_op, k=2, which="LA")
            
            # Transform data using eigenvectors
            Xv = self.X @ eigenvectors
            
            # Calculate ratio
            A = Xv.T @ (K @ Xv)
            ratio = np.diag(A) / np.diag(Xv.T @ Xv)
            ratios.append(ratio)
        
        # Stack all ratios into a matrix
        ratios = np.vstack(ratios)
        
        # Find first lambda where all ratios are below threshold
        tmp = np.where(np.apply_along_axis(np.sum, 1, ratios <= thresh) == ratios.shape[1])
        best_lambda = np.round(lambdas[np.min(tmp)], 4)
        
        # Store results for later visualization
        self.lambda_scores_ = {
            'lambdas': lambdas,
            'ratios': ratios,
            'method': 'original'
        }
        
        return best_lambda
     
    def calc_best_lambda_silhouette(self, lambda_range=(0, 5, 0.05), n_jobs=-1):
        """
        Compute best lambda using silhouette score
        """
        K = self.calkernel_()
        lambdas = np.arange(*lambda_range)
        scores = []
        
        for l in lambdas:
            # Always use implicit formation for consistency when use_implicit=True
            if self.use_implicit:
                adj_cov_op = self._compute_adjusted_covariance_implicit(self.X, K, l)
            else:
                adj_cov_op = CalAv(self.X, K, l)
            
            # Compute eigendecomposition
            _, eigenvectors = eigsh(adj_cov_op, k=self.n_components, which="LA")
            X_transformed = self.X @ eigenvectors
            
            batch_labels = self.confounding_matrix.argmax(axis=1)
            sil_score = silhouette_score(
                X_transformed, 
                batch_labels,
                n_jobs=n_jobs
            )
            scores.append(sil_score)
        
        best_idx = np.argmin(scores)
        best_lambda = np.round(lambdas[best_idx], 4)
        
        # Store lambda optimization results
        self.lambda_scores_ = {
            'lambdas': lambdas,
            'scores': np.array(scores),
            'method': 'silhouette'
        }
        
        return best_lambda
     
    def fit(self, X, y=None):
        """
        Fit AC-PCA model
        """
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        
        if self.preprocess:
            # Initialize preprocessors
            self.x_imputer_ = SimpleImputer(strategy='mean')
            self.x_scaler_ = StandardScaler(with_mean=self.center_x, 
                                          with_std=self.scale_x)
            self.y_imputer_ = SimpleImputer(strategy='mean')
            # Leave confounder scaling off by default; enabling it changes categorical IDs before make_confounder_matrix.
            self.y_scaler_ = StandardScaler(with_mean=self.center_y, 
                                          with_std=self.scale_y)
            
            # Preprocess X
            X = self.x_imputer_.fit_transform(X)
            self.X = self.x_scaler_.fit_transform(X)
            
            # Preprocess confounding factors
            y = y if y is not None else self.Y
            if y is not None:
                y = self.y_imputer_.fit_transform(y.reshape(-1, 1) if y.ndim == 1 else y)
                y = self.y_scaler_.fit_transform(y)
        else:
            self.X = X
        
        # Create confounding matrix and compute kernel
        self.confounding_matrix = self.make_confounder_matrix(y if y is not None else self.Y)
        K = self.calkernel_()
        
        # Calculate best lambda if needed
        if self.L == -1:
            if self.lambda_method == 'silhouette':
                self.L = self.calc_best_lambda_silhouette()
            elif self.lambda_method == 'original':
                self.L = self.calc_best_lambda()
            else:
                raise ValueError(f"Unknown lambda_method: {self.lambda_method}")
        
        # Always use implicit formation for consistency when use_implicit=True
        if self.use_implicit:
            adj_cov_op = self._compute_adjusted_covariance_implicit(self.X, K, self.L)
        else:
            adj_cov_op = CalAv(self.X, K, self.L)
        
        # Compute eigendecomposition
        self.e_, self.components_ = eigsh(adj_cov_op, 
                                        k=self.n_components, 
                                        which="LA")
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X
        """
        X = np.asarray(X)
        if self.preprocess:
            X = self.x_imputer_.transform(X)
            X = self.x_scaler_.transform(X)
        return X @ self.components_
    
    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply dimensionality reduction
        """
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space
        """
        X_scaled = X_transformed @ self.components_.T
        if self.preprocess:
            return self.x_scaler_.inverse_transform(X_scaled)
        return X_scaled
    
    def make_confounder_matrix(self, labels):
        """
        Generate matrix of confounding factors for ac-pca
        
        Parameters:
        -----------
        labels : array-like
            Batch labels (integers)
        
        Returns:
        --------
        y : array-like
            Confounding matrix
        """
        labels = np.asarray(labels).astype(int)  # Ensure integer labels
        ncol = np.max(labels) + 1
        nrow = labels.shape[0]
        
        pos_val = 1 / ncol * (ncol - 1)
        neg_val = -(1 - pos_val)
        
        y = np.zeros((nrow, ncol), dtype=np.float32)
        for i, index in enumerate(labels):
            y[i][index] = pos_val    
        y[np.where(y == 0.0)] = neg_val
        
        return y
     
    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'Y': self.Y,
            'L': self.L,
            'v': self.components_,
            'X': self.X,
            'lambda_method': self.lambda_method,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def plot_lambda_optimization(self):
        """
        Plot the relationship between lambda values and their scores
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'lambda_scores_'):
            raise ValueError("No lambda optimization results found. Run fit with L=-1 first.")
        
        plt.figure(figsize=(10, 6))
        
        if self.lambda_scores_['method'] == 'silhouette':
            plt.plot(self.lambda_scores_['lambdas'], self.lambda_scores_['scores'])
            plt.axvline(x=self.L, color='r', linestyle='--', label=f'Best λ = {self.L}')
            plt.ylabel('Silhouette Score')
            plt.title('Lambda Optimization (Silhouette Method)')
        else:  # original method
            plt.plot(self.lambda_scores_['lambdas'], 
                    np.mean(self.lambda_scores_['ratios'], axis=1))
            plt.axvline(x=self.L, color='r', linestyle='--', label=f'Best λ = {self.L}')
            plt.ylabel('Mean Ratio')
            plt.title('Lambda Optimization (Original Method)')
        
        plt.xlabel('Lambda')
        plt.legend()
        plt.grid(True)
        return plt

    def _compute_eigenpairs(self, operator, k):
        try:
            eigenvalues, eigenvectors = eigsh(
                operator, 
                k=k, 
                which="LA",
                tol=1e-10,  # Add tolerance
                maxiter=1000,  # Add max iterations
                return_eigenvectors=True
            )
            
            # Validate results
            if np.any(np.iscomplex(eigenvalues)):
                raise ValueError("Complex eigenvalues detected")
            
            # Ensure positive eigenvalues
            if np.any(eigenvalues <= 0):
                warnings.warn("Non-positive eigenvalues detected")
            
            return eigenvalues, eigenvectors
            
        except RuntimeError as e:
            raise RuntimeError(f"Eigendecomposition failed: {str(e)}")
