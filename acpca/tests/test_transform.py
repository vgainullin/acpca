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


def _mean_pairwise_distance(X, labels):
    total = 0.0
    groups = 0
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if idx.size < 2:
            continue
        vectors = X[idx]
        dists = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
        pairwise = dists[np.triu_indices(idx.size, k=1)]
        total += pairwise.mean()
        groups += 1
    return total / groups if groups else 0.0


def test_acpca_reduces_batch_structure_on_example_dataset():
    """AC-PCA should remove batch-driven structure while aligning biological replicates."""
    data = np.loadtxt('data/data_example1.csv', delimiter=',', skiprows=1)
    X = data[:, :-2]
    batch_labels = data[:, -2].astype(int)
    annotations = data[:, -1].astype(int)

    pca_components = PCA(n_components=2).fit_transform(X)

    acpca = ACPCA(n_components=2, L=1.0, preprocess=True)
    acpca.fit(X, batch_labels)
    acpca_components = acpca.transform(X)

    pca_batch_spread = _mean_pairwise_distance(pca_components, batch_labels)
    acpca_batch_spread = _mean_pairwise_distance(acpca_components, batch_labels)

    pca_annotation_spread = _mean_pairwise_distance(pca_components, annotations)
    acpca_annotation_spread = _mean_pairwise_distance(acpca_components, annotations)

    assert acpca_batch_spread > pca_batch_spread * 4, "Batch clustering persisted after AC-PCA"
    assert acpca_annotation_spread < pca_annotation_spread * 0.1, "Annotation replicates did not align after AC-PCA"


def _batch_centroids(embeddings, labels):
    sorted_labels = np.unique(labels)
    return np.vstack([embeddings[labels == label].mean(axis=0) for label in sorted_labels])


def test_acpca_zero_lambda_matches_pca_on_example_dataset():
    """With λ=0 and orientation alignment, AC-PCA should match PCA centroid layout."""
    data = np.loadtxt('data/data_example1.csv', delimiter=',', skiprows=1)
    X = data[:, :-2]
    batch_labels = data[:, -2].astype(int)
    annotations = data[:, -1].astype(int)

    pca_components = PCA(n_components=2).fit_transform(X)

    acpca = ACPCA(n_components=2, L=0.0, preprocess=True, align_orientation=True)
    acpca_components = acpca.fit_transform(X, batch_labels)

    # Match component signs to avoid trivial inversions
    for idx in range(pca_components.shape[1]):
        if np.dot(acpca_components[:, idx], pca_components[:, idx]) < 0:
            acpca_components[:, idx] *= -1

    pca_batch_spread = _mean_pairwise_distance(pca_components, batch_labels)
    acpca_batch_spread = _mean_pairwise_distance(acpca_components, batch_labels)

    pca_annotation_spread = _mean_pairwise_distance(pca_components, annotations)
    acpca_annotation_spread = _mean_pairwise_distance(acpca_components, annotations)

    assert_allclose(acpca_batch_spread, pca_batch_spread, rtol=1e-5)
    assert_allclose(acpca_annotation_spread, pca_annotation_spread, rtol=1e-5)

    pca_centroids = _batch_centroids(pca_components, batch_labels)
    acpca_centroids = _batch_centroids(acpca_components, batch_labels)
    assert_allclose(acpca_centroids, pca_centroids, atol=1e-6, rtol=1e-6)
