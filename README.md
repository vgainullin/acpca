# AC-PCA: Adjusted Confounding Principal Component Analysis

AC-PCA is a Python implementation of Adjusted Confounding Principal Component Analysis, a method designed to remove unwanted technical variation (batch effects) from high-dimensional data while preserving biological signal. It is particularly useful for analyzing high-dimensional biological data such as gene expression or methylation arrays where batch effects can obscure true biological signals.

## Features

- Removes batch effects while preserving biological variation
- Automatic lambda parameter selection using two different methods
- Visualization tools for lambda optimization
- Synthetic data generation for testing and validation
- Compatible with scikit-learn transformer interface

## Installation

### Using conda (recommended)

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/acpca.git
cd acpca
conda env create -f environment.yaml
conda activate acpca
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from acpca import ACPCA
from acpca.utils import create_synthetic_data

# Generate synthetic data
X, Y, batch_labels = create_synthetic_data(
    n_samples=300,
    n_genes=1000,
    num_batches=3,
    n_groups=2
)

# Initialize and fit AC-PCA
acpca = ACPCA(n_components=2, L=0.5)
acpca.fit(X, y=batch_labels)

# Transform data
X_transformed = acpca.transform(X)
```

## Detailed Usage

### Lambda Parameter Selection

The lambda parameter controls the trade-off between batch effect removal and preservation of biological signal. AC-PCA supports two methods for automatically selecting the optimal lambda:

1. Original Method (from the AC-PCA paper):
```python
acpca = ACPCA(n_components=2, L=-1, lambda_method='original')
acpca.fit(X, y=batch_labels)
```

2. Silhouette Score Method:
```python
acpca = ACPCA(n_components=2, L=-1, lambda_method='silhouette')
acpca.fit(X, y=batch_labels)
```

### Visualization

```python
# Fit AC-PCA with automatic lambda selection
acpca = ACPCA(n_components=2, L=-1)
acpca.fit(X, y=batch_labels)

# Plot lambda optimization results
acpca.plot_lambda_optimization()
```

## API Reference

### ACPCA Class

```python
ACPCA(Y=None, n_components=2, L=0.0, lambda_method='original')
```

Parameters:
- `Y`: array-like, optional - Confounding labels
- `n_components`: int - Number of components to keep
- `L`: float - Lambda value. If -1, best lambda will be calculated
- `lambda_method`: str - Method to calculate best lambda: 'original' or 'silhouette'

Methods:
- `fit(X, y=None)`: Fit the AC-PCA model
- `transform(X)`: Apply dimensionality reduction to X
- `plot_lambda_optimization()`: Visualize lambda selection process
- `get_params()`: Get parameters for this estimator
- `set_params(**params)`: Set the parameters of this estimator

### Utility Functions

```python
create_synthetic_data(
    n_samples=300,
    n_genes=1000,
    num_batches=3,
    n_groups=2,
    random_state=42
)
```

Generates synthetic gene expression data with batch effects and biological groups.

Parameters:
- `n_samples`: Number of samples (cells/specimens)
- `n_genes`: Number of genes (features)
- `num_batches`: Number of technical batches
- `n_groups`: Number of biological groups
- `random_state`: Random seed for reproducibility

## Examples

See the `notebooks/experiments.ipynb` notebook for detailed examples and visualizations, including:
- Basic usage with synthetic data
- Comparison of lambda selection methods
- Visualization of batch effect removal
- Real-world data examples

## Performance Tips

1. Scale your data before applying AC-PCA
2. Use the silhouette method for lambda selection with smaller datasets
3. Use the original method for larger datasets where computational efficiency is important
4. Consider reducing dimensionality with standard PCA before applying AC-PCA for very large datasets

## References

If you use this implementation in your research, please cite the original research paper:

```bibtex
@article{acpca2016,
  title={AC-PCA: simultaneous dimension reduction and adjustment for confounding variation bioRxiv},
  author={Z. Lin, C. Yang, Y. Zhu, J. C. Duchi, Y. Fu, Y. Wang, B. Jiang, M. Zamanighomi, X. Xu, M. Li, N. Sestan, H. Zhao, W. H. Wong},
  journal={bioRxiv},
  year={2016},
  doi={http://dx.doi.org/10.1101/040485}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

1. Clone your fork
2. Install development dependencies: `conda env create -f environment.yaml`
3. Install pre-commit hooks: `pre-commit install`
4. Run tests: `pytest tests/`

## Support

- Issue Tracker: [GitHub Issues](https://github.com/yourusername/acpca/issues)
- Documentation: See the `docs/` directory
