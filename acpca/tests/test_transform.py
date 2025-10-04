import numpy as np
from sklearn.decomposition import PCA
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from acpca import ACPCA

def test_acpca_equals_pca_when_lambda_zero():
    """
    Test that ACPCA with L=0.0 produces identical results to PCA
    """
    # Generate random data
    rng = np.random.RandomState(42)
    X = rng.randn(100, 20)
    n_components = 1
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(X)
    
    # Fit ACPCA with L=0.0
    acpca = ACPCA(n_components=n_components, Y=np.zeros((X.shape[0], )), L=0, preprocess=True, use_implicit=True, scale_x=False, scale_y=False, center_x=True, center_y=True)


    
    acpca_transformed = acpca.fit_transform(X)
    
    # Compare results (allowing for sign flips)
    for i in range(n_components):
        assert_allclose(
            np.abs(pca_transformed[:, i]),
            np.abs(acpca_transformed[:, i]),
            rtol=1e-10,
            err_msg=f"Component {i} differs between PCA and ACPCA(L=0)"
        )
    


def test_acpca_preprocess_preserves_confounder_labels():
    """Default preprocessing should not distort the confounder design matrix."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4))
    labels = np.array([0, 1, 2, 0, 1, 2])

    model = ACPCA(
        n_components=1,
        preprocess=True,
        use_implicit=True,
    )

    model.fit(X, labels)

    recovered = model.confounding_matrix.argmax(axis=1)
    assert_array_equal(
        recovered,
        labels,
        err_msg="Confounder preprocessing remapped batch labels",
    )
