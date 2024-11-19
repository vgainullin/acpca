import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def create_synthetic_data(n_samples=300, n_genes=1000, num_batches=3, n_groups=2, random_state=42):
    """
    Create synthetic gene expression data with batch effects.
    Ensures equal distribution of groups across batches to avoid any correlation.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples (cells/specimens)
    n_genes : int
        Number of genes (features)
    num_batches : int
        Number of technical batches
    n_groups : int
        Number of biological groups
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : array-like
        Expression matrix (n_samples x n_genes)
    Y : array-like
        Biological group labels
    batch_labels : array-like
        Batch assignments
    """
    np.random.seed(random_state)
    
    # Ensure n_samples is divisible by both n_groups and num_batches
    samples_per_group = n_samples // n_groups
    samples_per_batch = n_samples // num_batches
    
    # Create balanced group assignments
    Y = np.repeat(range(n_groups), samples_per_group)
    
    # Create balanced batch assignments ensuring equal distribution
    batch_labels = np.zeros(n_samples, dtype=int)
    for group in range(n_groups):
        group_mask = (Y == group)
        group_indices = np.where(group_mask)[0]
        
        # Assign equal number of samples from each group to each batch
        samples_per_group_batch = len(group_indices) // num_batches
        for batch in range(num_batches):
            start_idx = batch * samples_per_group_batch
            end_idx = (batch + 1) * samples_per_group_batch
            batch_labels[group_indices[start_idx:end_idx]] = batch
    
    # Shuffle the batch assignments within each group
    for group in range(n_groups):
        group_mask = (Y == group)
        group_indices = np.where(group_mask)[0]
        batch_labels[group_indices] = np.random.permutation(batch_labels[group_indices])
    
    # Initialize expression matrix
    X = np.zeros((n_samples, n_genes))
    
    # Base expression patterns for each biological group
    group_signatures = np.random.normal(0, 1, (n_groups, n_genes))
    
    # Generate expression data
    for i in range(n_samples):
        group = Y[i]
        batch = batch_labels[i]
        
        # Base expression for biological group
        base_expr = np.exp(group_signatures[group] + np.random.normal(0, 0.1, n_genes))
        
        # Add batch effects
        scale_factor = np.random.normal(1 + batch * 0.2, 0.1)
        dropout_prob = np.clip(0.7 + batch * 0.05, 0, 1)
        dropout_mask = np.random.binomial(1, dropout_prob, n_genes)
        tech_noise = np.random.normal(batch * 0.1, 0.2, n_genes)
        
        # Combine effects
        X[i] = base_expr * scale_factor * dropout_mask + tech_noise
    
    # Ensure non-negative values
    X = np.maximum(X, 0)
    
    # Verify batch-group distribution
    if __debug__:
        from pandas import crosstab
        from scipy.stats import chi2_contingency
        
        contingency = crosstab(batch_labels, Y)
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        print("\nBatch-Group distribution:")
        print(contingency)
        print(f"\nChi-square test p-value: {p_value:.4f}")
    
    return X, Y, batch_labels


def save_synthetic_data_to_csv(X, Y, batch_labels, filename='synthetic_data.csv'):
    """
    Save synthetic data to a CSV file
    """
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['label'] = Y
    df['batch'] = batch_labels
    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")

def plot_synthetic_data(df, Y, file_name, save_path='plots'):
    """
    Plot the synthetic data to visualize batch effects and save to file
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=Y, cmap='viridis')
    
    plt.title('PCA plot of synthetic data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_path, f'{file_name}.png'), dpi=300, bbox_inches='tight')
    plt.close() 