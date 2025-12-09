# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:19:30 2025

@author: yuyu2
"""


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import qr, svd
from matplotlib.patches import FancyArrowPatch

# Custom 2D arrow class
class Arrow2D(FancyArrowPatch):
    def __init__(self, origin, direction, *args, **kwargs):
        super().__init__(origin, direction, *args, **kwargs)


def compute_principal_angles(V_a_hat, V_b_hat):
    """
    Calculate principal angles between two subspaces (in degrees)
    :param V_a_hat: Normalized basis matrix of rank a (N x 2)
    :param V_b_hat: Normalized basis matrix of rank b (N x 2)
    :return: Two principal angles [theta1, theta2] (in degrees)
    """
    # Compute inner product matrix
    inner_product = V_a_hat.T @ V_b_hat  # Shape 2x2
    
    # SVD decomposition
    U, s, Vh = svd(inner_product)
    
    # s are singular values, i.e., cos(theta_i)
    angles_rad = np.arccos(np.clip(s, -1.0, 1.0))  # Avoid numerical errors
    angles_deg = np.degrees(angles_rad)
    return angles_deg

def compute_vaf_ratio(V_a_hat, V_b_hat, G_a):
    """
    Calculate VAF ratio between two subspaces
    :param V_a_hat: Normalized basis matrix of rank a (N x 2)
    :param V_b_hat: Normalized basis matrix of rank b (N x 2)
    :param G_a: Response matrix of rank a (N x 6)
    :return: VAF_ab ratio
    """
    # Project onto rank b subspace
    projection_matrix = V_b_hat @ V_b_hat.T  # Shape NxN
    G_a_projected = projection_matrix @ G_a  # Shape Nx6
    
    # Compute variance (Frobenius norm squared)
    var_original = np.linalg.norm(G_a, 'fro')**2
    var_projected = np.linalg.norm(G_a_projected, 'fro')**2
    
    vaf = var_projected / var_original
    return vaf

def gain_modulation_approximation(kappa_r):
    """
    Approximate κ(r, l) through gain modulation model, corrected version
    :param kappa_r: Input matrix, shape 2x6 (6 positions for each rank)
    :return: lambda_r, O_r, f_l, similarity_score
    """
    # Step 1: Singular Value Decomposition (SVD) of kappa_r
    U, s, Vh = svd(kappa_r, full_matrices=False)
    
    # Step 2: Rank-1 approximation using only the first singular value
    lambda_r = s[0]  # Maximum singular value as modulation factor
    O_r = U[:, 0].reshape(-1, 1)  # Extract first column as orthogonal basis (shape 2x1)
    f_l = Vh[0, :].reshape(1, -1)  # Extract first row as filling vector (shape 1x6)
    
    # Step 3: Compute approximated kappa_hat
    kappa_hat = O_r @ (lambda_r * f_l)  # Shape 2x6
    
    # Step 4: Compute error and similarity score
    error = kappa_r - kappa_hat
    norm_error = np.linalg.norm(error, 'fro')**2
    norm_kappa = np.linalg.norm(kappa_r, 'fro')**2
    similarity = 1 - (norm_error / norm_kappa)
    
    return lambda_r, O_r, f_l, similarity

def draw_angle(ax, vec1, vec2, color, label, radius=0.5):
    """Helper function to draw angles between vectors in 3D"""
    vec1_normalized = vec1 / np.linalg.norm(vec1)
    vec2_normalized = vec2 / np.linalg.norm(vec2)
    angle = np.arccos(np.dot(vec1_normalized, vec2_normalized))
    # Generate arc points
    n = 50
    arc_points = np.array([
        radius * np.cos(t) * vec1_normalized + radius * np.sin(t) * vec2_normalized
        for t in np.linspace(0, angle, n)
    ])
    ax.plot(arc_points[:,0], arc_points[:,1], arc_points[:,2], color=color, lw=2, label=label)

# File list placeholder - users should update with their own data paths
file_list = [
    'data/sample_dataset_1',  # Placeholder path 1
    'data/sample_dataset_2'   # Placeholder path 2
]

scaler = StandardScaler()

# Main processing loop
for day_n, file in enumerate(file_list):
    print(f"Processing file: {file}")
    
    file_list_temp = []
    data_len = []
    
    # Load data from two sessions
    for i in range(1, 3):
        file_temp = f'{file}_population_decoding_dataset_for_ses{i}.csv'
        df_temp = pd.read_csv(file_temp)
        # df_temp = df_temp[df_temp.label != '0']  # Optional filtering
        df_temp.label = str(i) + df_temp.label
        file_list_temp.append(df_temp)
        data_len.append(len(df_temp))
    
    # Combine datasets
    decoding_dataset_temp = pd.concat(file_list_temp, axis=0, join='inner', ignore_index=False)
    GLM_prepare = decoding_dataset_temp.iloc[:, 6:]
    
    # Create one-hot encoded labels for each condition
    for label in ['R', 'P', 'N1', 'N2']:
        for trial in range(1, 3):
            traillabel = f't{trial}d{label}'
            onehot_label = (decoding_dataset_temp.label == str(trial) + label).astype(int)
            GLM_prepare[traillabel] = np.array(onehot_label)
    
    beta_list_allrep = []
    
    # Multiple repetitions for robust beta estimation
    for rep_n in range(100):
        print(f"Repetition: {rep_n}")
        df_shuffled = GLM_prepare.sample(frac=1)  # Shuffle all data
        
        # Split into two halves
        half_size = len(df_shuffled) // 2
        df1 = df_shuffled.iloc[:half_size]
        df2 = df_shuffled.iloc[half_size:]
        units = df1.columns[:-8]
        
        for df in [df1, df2]:
            X = df.iloc[:, -8:]
            columns = X.columns
            X = np.array(X)
            beta_list = []
            
            # Fit Lasso regression for each neuron
            for column in range(GLM_prepare.shape[1] - 8):
                Y = df.iloc[:, column]
                Y = np.array(Y)
                
                if np.std(Y) == 0:
                    Y = Y
                else:
                    Y = (Y - np.mean(Y)) / np.std(Y)
                
                # Use LassoCV for regularized regression
                lasso = LassoCV(cv=5, max_iter=500)
                lasso.fit(X, Y)
                beta = lasso.coef_
                beta_list.append(beta)
            
            beta_array = np.array(beta_list)
            beta_df = pd.DataFrame(beta_array)
            beta_df.columns = columns
            beta_list_allrep.append(beta_df)
    
    # Combine all beta dataframes and compute mean
    combined_df = pd.concat(beta_list_allrep)
    mean_df = combined_df.groupby(combined_df.index).mean()
    mean_df.insert(0, column='neuron', value=units)
    mean_df['neuron'] = str(Path(file).name) + '_' + mean_df['neuron'].astype(str)
    mean_df.to_csv(f'{Path(file).name}_beta.csv')
    
    # Extract beta values for each condition
    beta_R = np.array(mean_df[['t1dR', 't2dR']]).T
    beta_P = np.array(mean_df[['t1dP', 't2dP']]).T
    beta_N1 = np.array(mean_df[['t1dN1', 't2dN1']]).T
    beta_N2 = np.array(mean_df[['t1dN2', 't2dN2']]).T
    
    N = mean_df.shape[0]
    
    # Demean beta values
    mean_beta_R = np.mean(beta_R, axis=0)
    mean_beta_P = np.mean(beta_P, axis=0)
    mean_beta_N1 = np.mean(beta_N1, axis=0)
    mean_beta_N2 = np.mean(beta_N2, axis=0)

    beta_R_demeaned = beta_R - mean_beta_R
    beta_P_demeaned = beta_P - mean_beta_P   
    beta_N1_demeaned = beta_N1 - mean_beta_N1
    beta_N2_demeaned = beta_N2 - mean_beta_N2
    
    # PCA for R condition
    pca = PCA(n_components=2)
    pca.fit(beta_R_demeaned)
    v_R_1 = pca.components_[0].reshape(-1, 1)
    v_R_2 = pca.components_[1].reshape(-1, 1)
    v_R_1 = v_R_1 / np.linalg.norm(v_R_1) * np.sqrt(N)  # Ensure norm is sqrt(N)
    v_R_2 = v_R_2 / np.linalg.norm(v_R_2) * np.sqrt(N)
    v_R = np.hstack([v_R_1, v_R_2])
    v_R_hat = v_R / np.sqrt(N)
    base_R = pca.transform(v_R_hat.T)
    kappa_R = pca.transform(beta_R_demeaned)
    
    # PCA for P condition
    pca = PCA(n_components=2)
    pca.fit(beta_P_demeaned)
    v_P_1 = pca.components_[0].reshape(-1, 1)
    v_P_2 = pca.components_[1].reshape(-1, 1)
    v_P_1 = v_P_1 / np.linalg.norm(v_P_1) * np.sqrt(N)
    v_P_2 = v_P_2 / np.linalg.norm(v_P_2) * np.sqrt(N)
    v_P = np.hstack([v_P_1, v_P_2])
    v_P_hat = v_P / np.sqrt(N)
    base_P = pca.transform(v_P_hat.T)
    kappa_P = pca.transform(beta_P_demeaned)
    
    # PCA for N1 condition
    pca = PCA(n_components=2)
    pca.fit(beta_N1_demeaned)
    v_N1_1 = pca.components_[0].reshape(-1, 1)
    v_N1_2 = pca.components_[1].reshape(-1, 1)
    v_N1_1 = v_N1_1 / np.linalg.norm(v_N1_1) * np.sqrt(N)
    v_N1_2 = v_N1_2 / np.linalg.norm(v_N1_2) * np.sqrt(N)
    v_N1 = np.hstack([v_N1_1, v_N1_2])
    v_N1_hat = v_N1 / np.sqrt(N)
    base_N1 = pca.transform(v_N1_hat.T)
    kappa_N1 = pca.transform(beta_N1_demeaned)
    
    # PCA for N2 condition
    pca = PCA(n_components=2)
    pca.fit(beta_N2_demeaned)
    v_N2_1 = pca.components_[0].reshape(-1, 1)
    v_N2_2 = pca.components_[1].reshape(-1, 1)
    v_N2_1 = v_N2_1 / np.linalg.norm(v_N2_1) * np.sqrt(N)
    v_N2_2 = v_N2_2 / np.linalg.norm(v_N2_2) * np.sqrt(N)
    v_N2 = np.hstack([v_N2_1, v_N2_2])
    v_N2_hat = v_N2 / np.sqrt(N)
    base_N2 = pca.transform(v_N2_hat.T)
    kappa_N2 = pca.transform(beta_N2_demeaned)
    
    # Compute principal angles between conditions
    # R vs P
    theta1, theta2 = compute_principal_angles(v_R_hat, v_P_hat)
    print(f"Principal angles between R and P: {theta1:.2f}°, {theta2:.2f}°")
    
    # Create 3D visualization of subspaces
    # Standard basis for subspace 1 (XY plane)
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    
    # Orthogonal basis construction for subspace 2
    v1 = np.array([np.cos(np.radians(theta1)), 0, np.sin(np.radians(theta1))])  # XZ plane tilt θ₁
    v2 = np.array([0, np.cos(np.radians(theta2)), np.sin(np.radians(theta2))])  # YZ plane tilt θ₂
    v2 = v2 - np.dot(v2, v1) * v1 / np.linalg.norm(v1)**2  # Orthogonalization
    
    # Verify orthogonality
    assert np.isclose(np.dot(v1, v2), 0, atol=1e-6), "Basis vectors must be orthogonal"
    
    # Projection data (example: 2 points per subspace)
    proj1 = kappa_R  # Points in subspace 1 (local coordinates)
    proj2 = kappa_P  # Points in subspace 2 (local coordinates)
    
    # Convert subspace 2 local coordinates to global 3D coordinates
    proj2_3d = np.array([c * v1 + d * v2 for c, d in proj2])
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot basis vectors
    ax.quiver(0, 0, 0, e1[0], e1[1], e1[2], color='k', lw=2, label='Subspace 1 Axes', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, e2[0], e2[1], e2[2], color='k', lw=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', lw=2, label='Subspace 2 Axes', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='r', lw=2, arrow_length_ratio=0.1)
    
    # Plot projection points
    ax.scatter(proj1[:,0], proj1[:,1], 0, c=['blue', 'green'], s=100, depthshade=False, 
               edgecolors='k', label='Projections on Subspace 1')
    ax.scatter(proj2_3d[:,0], proj2_3d[:,1], proj2_3d[:,2], c=['red', 'orange'], marker='s', 
               s=100, depthshade=False, edgecolors='k', label='Projections on Subspace 2')
    
    # Plot settings
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=-45)  # Initial view angle
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('3D Visualization of Subspaces R vs P')
    plt.tight_layout()
    plt.savefig(f'{Path(file).name}_3D_visualization_R_vs_P.svg')
    plt.show()
    
    # Compute and print VAF ratios
    G_R = v_R_hat @ kappa_R.T          
    vaf_ab = compute_vaf_ratio(v_R_hat, v_R_hat, G_R)
    print(f"VAF ratio (R to R): {vaf_ab:.4f}")
    
    vaf_ab = compute_vaf_ratio(v_R_hat, v_P_hat, G_R)
    print(f"VAF ratio (R to P): {vaf_ab:.4f}")
    
    # Compute gain modulation approximations
    lambda_r, O_r, f_l, similarity = gain_modulation_approximation(kappa_R)
    print(f"R space modulation factor λ_r: {lambda_r:.4f}")
    print(f"Orthogonal matrix O_r:\n{O_r}")
    print(f"Spatial position vector f_l (shape 2x6):\n{f_l.T}")
    print(f"Similarity score: {similarity:.4f}")
    
    # Save results to file
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f'{Path(file).name}_analysis_results.txt', 'w') as f:
        f.write(f"Analysis results for {file}\n")
        f.write("="*50 + "\n\n")
        
        f.write("PRINCIPAL ANGLES:\n")
        f.write(f"R vs P: {theta1:.2f}°, {theta2:.2f}°\n\n")
        
        f.write("VAF RATIOS:\n")
        vaf_ab = compute_vaf_ratio(v_R_hat, v_R_hat, G_R)
        f.write(f"R to R: {vaf_ab:.4f}\n")
        vaf_ab = compute_vaf_ratio(v_R_hat, v_P_hat, G_R)
        f.write(f"R to P: {vaf_ab:.4f}\n\n")
        
        f.write("GAIN MODULATION APPROXIMATION:\n")
        f.write(f"R space modulation factor λ_r: {lambda_r:.4f}\n")
        f.write(f"Orthogonal matrix O_r:\n{O_r}\n")
        f.write(f"Spatial position vector f_l:\n{f_l.T}\n")
        f.write(f"Similarity score: {similarity:.4f}\n")

print("Analysis completed successfully!")