import numpy as np
from sklearn.decomposition import PCA
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from acpca import ACPCA

_EXPECTED_ALIGNMENT_ANGLE_DEG = 90.0

_EXPECTED_L1_CENTROIDS = np.array([
    [-9.83726625e+00, -2.97068065e+01],
    [-3.72844079e+00, -2.39200008e+01],
    [ 2.13054839e+00, -1.75134593e+01],
    [ 5.27164407e+00, -1.09065232e+01],
    [ 6.28521626e+00, -4.36497129e+00],
    [ 6.17325522e+00,  2.43526888e+00],
    [ 3.99268907e+00,  9.37588033e+00],
    [ 3.82476410e-01,  1.75488565e+01],
    [-3.20079565e+00,  2.51287564e+01],
    [-7.46932674e+00,  3.19229990e+01],
])

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



def test_acpca_align_orientation_preserves_geometry():
    """Alignment flag should only rotate embeddings for λ>0."""
    data = np.loadtxt('data/data_example1.csv', delimiter=',', skiprows=1)
    X = data[:, :-2]
    batch_labels = data[:, -2].astype(int)

    base_model = ACPCA(
        n_components=2,
        L=1.0,
        preprocess=True,
        use_implicit=False,
        align_orientation=False,
    )
    base_coords = base_model.fit_transform(X, batch_labels)

    aligned_model = ACPCA(
        n_components=2,
        L=1.0,
        preprocess=True,
        use_implicit=False,
        align_orientation=True,
    )
    aligned_coords = aligned_model.fit_transform(X, batch_labels)

    rotation, aligned_centered, base_centered = _solve_rotation(aligned_coords, base_coords)

    assert_allclose(rotation.T @ rotation, np.eye(2), atol=1e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)

    aligned_recovered = aligned_centered @ rotation
    assert_allclose(aligned_recovered, base_centered, atol=1e-5)

    gram_aligned = aligned_centered @ aligned_centered.T
    gram_base = base_centered @ base_centered.T
    assert_allclose(gram_aligned, gram_base, atol=1e-4, rtol=1e-6)

    angle = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    angle = (angle + 360) % 360
    alt_angle = (360 - _EXPECTED_ALIGNMENT_ANGLE_DEG) % 360
    angle_error = min(abs(angle - _EXPECTED_ALIGNMENT_ANGLE_DEG), abs(angle - alt_angle))
    assert angle_error < 1e-3


def _match_signs(matrix, reference):
    aligned = matrix.copy()
    for idx in range(aligned.shape[1]):
        if np.dot(aligned[:, idx], reference[:, idx]) < 0:
            aligned[:, idx] *= -1
    return aligned


def _solve_rotation(source, target):
    src_centered = source - source.mean(axis=0, keepdims=True)
    tgt_centered = target - target.mean(axis=0, keepdims=True)
    u_align, _, vt_align = np.linalg.svd(src_centered.T @ tgt_centered, full_matrices=False)
    rotation = u_align @ vt_align
    return rotation, src_centered, tgt_centered


def test_acpca_lambda_one_centroids_baseline():
    """Recorded centroids for λ=1 without alignment remain stable up to sign."""
    data = np.loadtxt('data/data_example1.csv', delimiter=',', skiprows=1)
    X = data[:, :-2]
    batch_labels = data[:, -2].astype(int)
    annotations = data[:, -1].astype(int)

    model = ACPCA(
        n_components=2,
        Y=batch_labels,
        L=1.0,
        preprocess=True,
        center_x=True,
        align_orientation=False,
        use_implicit=False,
    )
    centroids = _batch_centroids(model.fit_transform(X, batch_labels), annotations)
    centroids = _match_signs(centroids, _EXPECTED_L1_CENTROIDS)
    assert_allclose(centroids, _EXPECTED_L1_CENTROIDS, atol=1e-6, rtol=1e-6)



def test_acpca_lambda_one_alignment_centroid_shape_stable():
    """Aligned centroids retain the expected number of batches and components."""
    data = np.loadtxt('data/data_example1.csv', delimiter=',', skiprows=1)
    X = data[:, :-2]
    batch_labels = data[:, -2].astype(int)
    annotations = data[:, -1].astype(int)

    aligned_model = ACPCA(
        n_components=2,
        Y=batch_labels,
        L=1.0,
        preprocess=True,
        center_x=True,
        align_orientation=True,
        use_implicit=False,
    )
    aligned_centroids = _batch_centroids(aligned_model.fit_transform(X, batch_labels), annotations)

    assert aligned_centroids.shape == _EXPECTED_L1_CENTROIDS.shape


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
