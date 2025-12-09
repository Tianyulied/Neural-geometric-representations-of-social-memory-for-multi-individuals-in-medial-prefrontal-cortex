# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:53:32 2025

@author: yuyu2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import mannwhitneyu, ttest_rel
from scipy.linalg import svd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold


def joint_svd_projection(datasets, n_components=8):
    """
    Perform joint dimensionality reduction using SVD
    
    Parameters:
    -----------
    datasets : list of numpy arrays
        List of datasets, each of shape (n_neurons, 8)
    n_components : int
        Number of components to keep
        
    Returns:
    --------
    numpy array
        Projected data for each dataset, shape (n_datasets, 8, n_components)
    """
    # Stack all neuron responses (neurons × stimuli)
    stacked_center = [d - d.mean() for d in datasets]
    X_combined = np.vstack(stacked_center)  # Combined shape (total_neurons × 8)
    
    # Center the data (by stimulus dimension)
    X_centered = X_combined - X_combined.mean(axis=0, keepdims=True)
    
    # Perform joint SVD (dimensionality reduction in neuron direction)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)  # U: (total_neurons × 8)
    
    # Split and project to each dataset
    projections = []
    start = 0
    
    for data in datasets:
        n_neurons = data.shape[0]
        end = start + n_neurons
        
        # Extract U components for this dataset (neurons × n_components)
        U_sub = U[start:end, :n_components]
        
        # Perform QR decomposition for orthonormal basis
        Q, R = np.linalg.qr(U_sub, mode='reduced')
        
        # Ensure consistent orientation (positive correlation with original)
        for pc in range(n_components):
            cc = np.corrcoef(U_sub[:, pc], Q[:, pc])
            if cc[0, 1] < 0:
                Q[:, pc] = -Q[:, pc]
        
        # Compute projection (stimuli × components)
        proj = (data - data.mean(axis=0)).T @ Q  # Shape (8, n_components)
        projections.append(proj.T)  # Transpose to (n_components × 8)
        start = end
    
    return np.array(projections)


def principal_angles(Q1, Q2):
    """
    Calculate principal angles between two subspaces (in degrees)
    
    Parameters:
    -----------
    Q1, Q2 : numpy arrays
        Orthonormal basis matrices, shape (m, k)
        
    Returns:
    --------
    angles : numpy array
        Principal angles in degrees
    cosines : numpy array
        Corresponding cosine values
    """
    # Verify orthogonality of input matrices
    assert np.allclose(Q1.T @ Q1, np.eye(Q1.shape[1])), "Q1 is not an orthonormal basis matrix"
    assert np.allclose(Q2.T @ Q2, np.eye(Q2.shape[1])), "Q2 is not an orthonormal basis matrix"
    
    # Compute product matrix
    M = Q1.T @ Q2
    
    # Compute singular values
    _, s, _ = svd(M)
    
    # Avoid numerical errors that could cause cosine values outside [-1, 1]
    cosines = np.clip(s, -1.0, 1.0)
    
    # Compute angles (convert to degrees)
    angles = np.degrees(np.arccos(cosines))
    
    return angles, cosines


def compute_vaf_ratio(V_a_hat, V_b_hat, G_a):
    """
    Calculate VAF ratio between two subspaces
    
    Parameters:
    -----------
    V_a_hat : numpy array
        Normalized basis matrix of rank a (N × 2)
    V_b_hat : numpy array
        Normalized basis matrix of rank b (N × 2)
    G_a : numpy array
        Response matrix of rank a (N × 6)
        
    Returns:
    --------
    float
        VAF_ab ratio
    """
    # Project onto rank b subspace
    projection_matrix = V_b_hat @ V_b_hat.T  # Shape N×N
    G_a_projected = projection_matrix @ G_a  # Shape N×6
    
    # Compute variance (Frobenius norm squared)
    var_original = np.linalg.norm(G_a, 'fro')**2
    var_projected = np.linalg.norm(G_a_projected, 'fro')**2
    
    vaf = var_projected / var_original
    return vaf


def gain_modulation_approximation(kappa_r):
    """
    Approximate κ(r, l) through gain modulation model
    
    Parameters:
    -----------
    kappa_r : numpy array
        Input matrix, shape (2, 6) (6 positions for each rank)
        
    Returns:
    --------
    lambda_r : float
        Modulation factor
    O_r : numpy array
        Orthogonal basis matrix
    f_l : numpy array
        Filling vector
    similarity : float
        Similarity score
    """
    # Step 1: Singular Value Decomposition (SVD) of kappa_r
    U, s, Vh = svd(kappa_r, full_matrices=False)
    
    # Step 2: Rank-1 approximation using only the first singular value
    lambda_r = s[0]  # Maximum singular value as modulation factor
    O_r = U[:, 0].reshape(-1, 1)  # Extract first column as orthogonal basis (shape 2×1)
    f_l = Vh[0, :].reshape(1, -1)  # Extract first row as filling vector (shape 1×6)
    
    # Step 3: Compute approximated kappa_hat
    kappa_hat = O_r @ (lambda_r * f_l)  # Shape 2×6
    
    # Step 4: Compute error and similarity score
    error = kappa_r - kappa_hat
    norm_error = np.linalg.norm(error, 'fro')**2
    norm_kappa = np.linalg.norm(kappa_r, 'fro')**2
    similarity = 1 - (norm_error / norm_kappa)
    
    return lambda_r, O_r, f_l, similarity


def projection_distance(A, B):
    """
    Calculate Frobenius distance between two subspaces' projections
    
    Parameters:
    -----------
    A, B : numpy arrays
        Basis matrices for subspaces
        
    Returns:
    --------
    float
        Frobenius distance between projection matrices
    """
    Pa = A @ A.T
    Pb = B @ B.T
    return np.linalg.norm(Pa - Pb, 'fro')


def compute_similarity(results):
    """
    Compute cosine similarity matrix from SVD results
    
    Parameters:
    -----------
    results : numpy array
        Results from joint SVD, shape (n_datasets, 8, n_components)
        
    Returns:
    --------
    numpy array
        Similarity matrix, shape (n_datasets, n_datasets)
    """
    n_datasets = results.shape[0]
    sim_matrix = np.zeros((n_datasets, n_datasets))
    
    for i in range(n_datasets):
        for j in range(n_datasets):
            # Compute average cosine similarity across all stimuli
            total_sim = 0
            for stim in range(8):
                vec_i = results[i, stim, :]
                vec_j = results[j, stim, :]
                total_sim += cosine_similarity([vec_i], [vec_j])[0][0]
            sim_matrix[i][j] = total_sim / 8
    
    return sim_matrix


def cv_mse(X_data, y_data, alpha, n_splits=10):
    """
    Compute cross-validated mean squared error for Lasso regression
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred)**2)
        mse_list.append(mse)
    
    return np.array(mse_list)


def cv_metrics(X_data, y_data, alpha, n_splits=10):
    """
    Compute cross-validated MSE and R² for Lasso regression
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    r2_list = []
    
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred)**2)
        mse_list.append(mse)
        
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
    
    return np.array(mse_list), np.array(r2_list)


def process_iteration(iter_num, file):
    """
    Process a single iteration of shuffled data analysis
    
    Note: Users should implement their own data loading and processing
    """
    # Placeholder for data processing - users should implement based on their data structure
    print(f"Processing iteration {iter_num} for file {file}")
    
    # Example implementation - users should replace with actual data loading
    n_samples = 100
    n_neurons = 50
    synthetic_data = np.random.randn(n_samples, n_neurons)
    
    # Create synthetic beta coefficients
    beta_list = np.random.randn(n_neurons, 8)
    
    return beta_list


def main():
    """Main analysis function"""
    # Set up paths and parameters
    home_file = Path('data/analysis')  # Example path - users should update
    
    # Example dataset groups - users should update with their own data
    test_list = [
        ['sample_dataset_1_day1', 'sample_dataset_1_day2', 'sample_dataset_1_day3'],
        ['sample_dataset_2_day1', 'sample_dataset_2_day2', 'sample_dataset_2_day3']
    ]
    
    # Create results directory
    results_dir = home_file / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results
    vaf_different_space = []
    vaf_in_specific_space = []
    vaf_p_list = []
    vaf_p_list_R = []
    vaf_p_list_P = []
    vaf_p_list_N1 = []
    vaf_p_list_N2 = []
    
    # Process each group of datasets
    for mouse_list in test_list:
        print(f"Processing dataset group: {mouse_list}")
        
        # Load datasets
        datasets = []
        for mouse in mouse_list:
            # Example: Load beta coefficients from CSV
            # df_temp = pd.read_csv(home_file / f'{mouse}_beta.csv')
            # datasets.append(np.array(df_temp.iloc[:, 2:]))
            
            # For demonstration, create synthetic data
            n_neurons = 50
            synthetic_betas = np.random.randn(n_neurons, 8)
            datasets.append(synthetic_betas)
        
        # Perform joint dimensionality reduction
        svd_results = joint_svd_projection(datasets, n_components=8)
        print(f"Joint SVD results shape: {svd_results.shape}")
        
        # Compute and visualize similarity matrix
        similarity = compute_similarity(svd_results)
        similarity = np.round(similarity, 2)
        
        plt.figure(figsize=(8, 6))
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.family'] = 'Times New Roman'
        
        sns.heatmap(similarity,
                    fmt=".2f",
                    cmap="jet",
                    vmin=0, vmax=1,
                    xticklabels=[f'day{i+1}' for i in range(similarity.shape[0])],
                    yticklabels=[f'day{i+1}' for i in range(similarity.shape[0])],
                    square=True)
        
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                plt.text(j + 0.5, i + 0.5, str(similarity[i, j]),
                         ha='center', va='center', color='black')
        
        plt.title("Joint Similarity Matrix")
        plt.xlabel("Dataset Index")
        plt.ylabel("Dataset Index")
        plt.savefig(results_dir / f'{mouse_list[0]}_joint_similarity.svg')
        plt.savefig(results_dir / f'{mouse_list[0]}_joint_similarity.png')
        plt.show()
        
        # Store true similarity for later comparison
        true_similarity = similarity.copy()
        
        # Analyze subspaces for each condition
        subspace_dict = {}
        kappa_dict = {}
        angle_true_dict = {}
        cosine_true_dict = {}
        vaf_true_dict = {}
        lambda_r_df = pd.DataFrame()
        
        # Process each social condition
        for demo_i, space_name in enumerate(['R', 'P', 'N1', 'N2']):
            print(f"Analyzing condition: {space_name}")
            
            # Extract data for this condition
            dataset_this_space = svd_results[:, :, 2*demo_i:2*demo_i+2]
            v_hat_list = []
            kappa_list = []
            
            # Process each day
            for day_idx in range(dataset_this_space.shape[0]):
                dataset_this_day = dataset_this_space[day_idx, :, :]
                
                # Perform PCA on this day's data
                beta_this_day = dataset_this_day.T  # Transpose to (2, 8)
                beta_demeaned = beta_this_day - beta_this_day.mean(axis=0)
                
                pca = PCA(n_components=2)
                pca.fit(beta_demeaned)
                
                # Extract principal components
                v1 = pca.components_[0].reshape(-1, 1)
                v2 = pca.components_[1].reshape(-1, 1)
                
                # Normalize
                N = 8
                v1 = v1 / np.linalg.norm(v1) * np.sqrt(N)
                v2 = v2 / np.linalg.norm(v2) * np.sqrt(N)
                
                v_matrix = np.hstack([v1, v2])
                v_hat = v_matrix / np.sqrt(N)
                
                # Project data
                kappa = pca.transform(beta_demeaned)
                
                # Store results
                v_hat_list.append(v_hat)
                kappa_list.append(kappa)
                subspace_dict[(f'space_{space_name}', f'day_{day_idx}')] = v_hat
                kappa_dict[(f'space_{space_name}', f'day_{day_idx}')] = kappa
                
                # Compute gain modulation approximation
                lambda_r, O_r, f_l, sim_score = gain_modulation_approximation(kappa)
                
                # Store lambda_r
                if f'space_{space_name}' not in lambda_r_df.columns:
                    lambda_r_df[f'space_{space_name}'] = np.nan
                lambda_r_df.at[day_idx, f'space_{space_name}'] = lambda_r
            
            # Compute principal angles and cosines
            n_days = len(v_hat_list)
            angles_matrix = np.zeros((n_days, n_days))
            cosine_matrix = np.zeros((n_days, n_days))
            vaf_matrix = np.zeros((n_days, n_days))
            
            for i in range(n_days):
                G_i = v_hat_list[i] @ kappa_list[i].T
                
                for j in range(n_days):
                    if i <= j:
                        angles, cosines = principal_angles(v_hat_list[i], v_hat_list[j])
                        angles_matrix[i, j] = angles[0]
                        angles_matrix[j, i] = angles[0]
                        cosine_matrix[i, j] = cosines[0]
                        cosine_matrix[j, i] = cosines[0]
                    
                    # Compute VAF
                    vaf = compute_vaf_ratio(v_hat_list[i], v_hat_list[j], G_i)
                    vaf_matrix[i, j] = vaf
            
            # Store results for this condition
            angle_true_dict[space_name] = angles_matrix
            cosine_true_dict[space_name] = cosine_matrix
            vaf_true_dict[space_name] = vaf_matrix
            vaf_in_specific_space.append(vaf_matrix)
            
            # Visualize cosine similarity
            cosine_matrix_rounded = np.round(cosine_matrix, 2)
            plt.figure(figsize=(8, 6))
            plt.rcParams['svg.fonttype'] = 'none'
            plt.rcParams['font.family'] = 'Times New Roman'
            
            sns.heatmap(cosine_matrix_rounded,
                        fmt=".2f",
                        cmap="jet",
                        vmin=0.8, vmax=1,
                        xticklabels=[f'day{i+1}' for i in range(n_days)],
                        yticklabels=[f'day{i+1}' for i in range(n_days)],
                        square=True)
            
            for i in range(n_days):
                for j in range(n_days):
                    plt.text(j + 0.5, i + 0.5, str(cosine_matrix_rounded[i, j]),
                             ha='center', va='center', color='black')
            
            plt.title(f"Cosine Similarity - Condition: {space_name}")
            plt.xlabel("Day")
            plt.ylabel("Day")
            plt.savefig(results_dir / f'{mouse_list[0]}_cosine_{space_name}.svg')
            plt.show()
        
        # Save lambda_r values
        lambda_r_df.to_csv(results_dir / f'{mouse_list[0]}_lambda_r.csv')
        
        # Analyze subspace distances
        keys = list(subspace_dict.keys())
        n_keys = len(keys)
        
        # Compute projection distances
        distance_matrix = np.zeros((n_keys, n_keys))
        for i in range(n_keys):
            for j in range(i+1, n_keys):
                dist = projection_distance(subspace_dict[keys[i]], subspace_dict[keys[j]])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Group distances by type
        within_stim_distances = []
        within_day_distances = []
        
        for i in range(n_keys):
            stim_i, day_i = keys[i]
            for j in range(i+1, n_keys):
                stim_j, day_j = keys[j]
                dist = distance_matrix[i, j]
                
                # Same stimulus across different days
                if stim_i == stim_j and day_i != day_j:
                    within_stim_distances.append(dist)
                
                # Same day across different stimuli
                if day_i == day_j and stim_i != stim_j:
                    within_day_distances.append(dist)
        
        # Statistical comparison
        if within_stim_distances and within_day_distances:
            mean_within_stim = np.mean(within_stim_distances)
            mean_within_day = np.mean(within_day_distances)
            
            stat, p_value = mannwhitneyu(within_stim_distances, within_day_distances)
            
            # Visualize comparison
            plt.figure(figsize=(10, 6))
            plt.rcParams['svg.fonttype'] = 'none'
            plt.rcParams['font.family'] = 'Times New Roman'
            
            plt.boxplot([within_stim_distances, within_day_distances],
                       labels=['Same Stimulus\nAcross Days', 'Same Day\nAcross Stimuli'])
            
            plt.title(f'Subspace Stability Comparison\n'
                     f'Within-stimulus distance: {mean_within_stim:.3f}\n'
                     f'Within-day distance: {mean_within_day:.3f}\n'
                     f'p = {p_value:.4f}')
            plt.ylabel('Projection Frobenius Distance')
            plt.grid(True)
            plt.savefig(results_dir / f'{mouse_list[0]}_subspace_comparison.svg')
            plt.show()
        
        # Analyze VAF between different spaces within the same day
        spaces = ['R', 'P', 'N1', 'N2']
        days = list(set([key[1] for key in keys]))
        
        vaf_different_space_all = []
        for day in days:
            vaf_matrix_day = np.zeros((len(spaces), len(spaces)))
            
            for i, space_i in enumerate(spaces):
                for j, space_j in enumerate(spaces):
                    key1 = (f'space_{space_i}', day)
                    key2 = (f'space_{space_j}', day)
                    
                    if key1 in vaf_true_dict and key2 in vaf_true_dict:
                        # This is a simplification - actual implementation would need to
                        # extract the appropriate VAF value from precomputed matrices
                        vaf_different_space_all.append(vaf_true_dict[space_i][i, j])
        
        if vaf_different_space_all:
            vaf_different_space.append(np.mean(vaf_different_space_all))
    
    # Summary visualization
    if vaf_different_space:
        plt.figure(figsize=(8, 6))
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.family'] = 'Times New Roman'
        
        plt.hist(vaf_different_space, bins=20, alpha=0.7)
        plt.title('VAF Between Different Spaces (All Datasets)')
        plt.xlabel('VAF Ratio')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(results_dir / 'vaf_between_spaces_summary.svg')
        plt.show()
    
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()