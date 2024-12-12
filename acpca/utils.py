import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def create_synthetic_data(n_samples=300, n_genes=1000, num_batches=3, n_groups=2, 
                         apply_scale_effect=True,
                         apply_dropout_effect=True,
                         apply_technical_noise=True,
                         apply_batch_structure=True,
                         random_state=42):
    """
    Create synthetic gene expression data with configurable batch effects.
    
    Returns:
    --------
    X : ndarray
        Expression data with batch effects
    X_true : ndarray
        True expression data without batch effects
    Y : ndarray
        Group labels
    batch_labels : ndarray
        Batch assignments
    """
    np.random.seed(random_state)
    
    # Ensure n_samples is divisible by both n_groups and num_batches
    samples_per_group = n_samples // n_groups
    
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
    
    # Initialize expression matrices
    X = np.zeros((n_samples, n_genes))
    X_true = np.zeros((n_samples, n_genes))
    
    # Make biological differences more subtle and sparse
    # Only a small subset of genes will be differentially expressed
    n_de_genes = int(n_genes * 0.05)  # Only 5% of genes are differential
    de_gene_indices = np.random.choice(n_genes, n_de_genes, replace=False)
    
    # Create more subtle group signatures
    group_signatures = np.zeros((n_groups, n_genes))
    group_signatures[:, de_gene_indices] = np.random.normal(0, 0.3, (n_groups, n_de_genes))
    
    # Generate expression data with configurable batch effects
    for i in range(n_samples):
        group = Y[i]
        batch = batch_labels[i]
        
        # Generate subtle biological signal
        biological_noise = np.random.normal(0, 0.05, n_genes)
        base_expr = np.exp(group_signatures[group] + biological_noise)
        
        # Store true expression without batch effects
        X_true[i] = base_expr
        
        # Initialize with base expression
        X[i] = base_expr
        
        if apply_scale_effect:
            scale_factor = np.random.normal(1 + batch * 1.0, 0.3)
            X[i] *= scale_factor
        
        if apply_dropout_effect:
            dropout_prob = np.clip(0.6 + batch * 0.15, 0, 0.9)
            dropout_mask = np.random.binomial(1, 1-dropout_prob, n_genes)
            X[i] *= dropout_mask
        
        if apply_technical_noise:
            tech_noise = np.random.normal(batch * 0.5, 0.5, n_genes)
            X[i] += tech_noise
            
            # Add batch-specific gene patterns
            batch_specific_pattern = np.zeros(n_genes)
            batch_genes = np.random.choice(n_genes, int(n_genes * 0.1), replace=False)
            batch_specific_pattern[batch_genes] = np.random.normal(batch * 0.5, 0.3)
            X[i] += batch_specific_pattern

    # Ensure non-negative values
    X = np.maximum(X, 0)
    
    if apply_batch_structure:
        # Add correlation between genes
        n_factors = 50
        correlation_matrix = np.random.normal(0, 1, (n_genes, n_factors))
        latent_factors = np.random.normal(0, 1, (n_factors, n_samples))
        batch_structure = correlation_matrix @ latent_factors
        
        batch_scaling_factors = np.linspace(0.5, 1.5, num_batches)
        batch_scaling = batch_scaling_factors[batch_labels]
        X += batch_structure.T * batch_scaling[:, None] * 0.5

    # Verify batch-group distribution
    if __debug__:
        from pandas import crosstab
        from scipy.stats import chi2_contingency
        
        contingency = crosstab(batch_labels, Y)
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        print("\nBatch-Group distribution:")
        print(contingency)
        print(f"\nChi-square test p-value: {p_value:.4f}")
        print("(High p-value indicates good balance between batches and groups)")
    
    return X, X_true, Y, batch_labels


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