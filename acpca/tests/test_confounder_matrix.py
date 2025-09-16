import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from acpca import ACPCA


def test_confounder_matrix_basic_construction():
    """Test basic matrix shape and centering property."""
    labels = np.array([0, 1, 2, 0, 1, 2])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    assert matrix.shape == (6, 3)
    row_sums = matrix.sum(axis=1)
    assert_allclose(row_sums, 0, atol=1e-10)


def test_confounder_matrix_identical_batches():
    """Test samples from same batch have identical rows."""
    labels = np.array([0, 0, 1, 1, 2, 2])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    assert_allclose(matrix[0], matrix[1])
    assert_allclose(matrix[2], matrix[3])
    assert_allclose(matrix[4], matrix[5])


def test_confounder_matrix_values_two_batches():
    """Test correct values for 2-batch case."""
    labels = np.array([0, 1])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    assert np.isclose(matrix[0, 0], 0.5)
    assert np.isclose(matrix[0, 1], -0.5)
    assert np.isclose(matrix[1, 0], -0.5)
    assert np.isclose(matrix[1, 1], 0.5)


def test_confounder_matrix_values_three_batches():
    """Test correct values for 3-batch case."""
    labels = np.array([0, 1, 2])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    expected_pos = 2.0/3.0
    expected_neg = -1.0/3.0

    assert np.isclose(matrix[0, 0], expected_pos)
    assert np.isclose(matrix[0, 1], expected_neg)
    assert np.isclose(matrix[0, 2], expected_neg)


def test_confounder_matrix_single_batch():
    """Test edge case with single batch."""
    labels = np.array([0, 0, 0])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    expected_matrix = np.array([[-1.0], [-1.0], [-1.0]], dtype=np.float32)
    assert_allclose(matrix, expected_matrix)
    assert_allclose(matrix[0], matrix[1])
    assert_allclose(matrix[1], matrix[2])


def test_confounder_matrix_non_sequential_labels():
    """Test with non-sequential batch labels."""
    labels = np.array([0, 2, 0, 2])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    assert matrix.shape == (4, 3)
    row_sums = matrix.sum(axis=1)
    assert_allclose(row_sums, 0, atol=1e-10)


def test_confounder_matrix_dtype():
    """Test matrix data type."""
    labels = np.array([0, 1, 2])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    assert matrix.dtype == np.float32


def test_confounder_matrix_with_large_labels():
    """Test with large label values."""
    labels = np.array([10, 15, 20])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    assert matrix.shape == (3, 21)
    row_sums = matrix.sum(axis=1)
    assert_allclose(row_sums, 0, atol=1e-6)


def test_confounder_matrix_argmax_consistency():
    """Test argmax recovers original labels."""
    labels = np.array([0, 1, 2, 1, 0, 2])
    acpca = ACPCA()
    matrix = acpca.make_confounder_matrix(labels)

    recovered_labels = matrix.argmax(axis=1)
    assert_array_equal(recovered_labels, labels)