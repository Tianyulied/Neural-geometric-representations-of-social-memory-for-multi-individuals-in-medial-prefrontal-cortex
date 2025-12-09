# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 17:34:37 2025

@author: yuyu2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr

# File list placeholder - users should update with their own data paths
file_list = [
    'data/sample_dataset_1',  # Placeholder path 1
    'data/sample_dataset_2'   # Placeholder path 2
]

#%% Ablation study function
def ablation_study(X_train, X_test, y_train, y_test, max_components):
    """Gradually increase PCA components and observe classification performance"""
    n_components = X_train.shape[1]
    max_components = min(n_components, max_components)
    
    svm_scores = []
    lr_scores = []
    
    for k in range(1, max_components + 1):
        # SVM classifier
        svm = SVC(
            kernel='linear', 
            class_weight='balanced',
            random_state=42
        ).fit(X_train[:, :k], y_train)
        svm_pred = svm.predict(X_test[:, :k])
        svm_scores.append(accuracy_score(y_test, svm_pred))
        
        # Logistic Regression classifier
        lr = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ).fit(X_train[:, :k], y_train)
        lr_pred = lr.predict(X_test[:, :k])
        lr_scores.append(accuracy_score(y_test, lr_pred))
        
        print(f"Components: {k:2d} | SVM Acc: {svm_scores[-1]:.3f} | LR Acc: {lr_scores[-1]:.3f}")

    return svm_scores, lr_scores

#%% Baseline model comparison
def get_contributions(model, pca):
    """Calculate feature contributions for different models"""
    if isinstance(model, SVC):
        weights = np.mean(np.abs(model.coef_), axis=0)
    elif isinstance(model, LogisticRegression):
        weights = np.mean(np.abs(model.coef_), axis=0)
    else:
        raise ValueError("Unsupported model type")
    return weights * pca.explained_variance_ratio_

#%% Main processing loop
for file in file_list:
    # Load data (placeholder - users should implement their own data loading)
    # Example: df_temp = pd.read_csv(f'{file}.csv')
    print(f"Processing: {file}")
    
    # Placeholder for data loading - users should replace this
    # y = df_temp['label']
    # X = df_temp.drop(['label'], axis=1)
    
    # Generate sample data for demonstration
    X, y = make_classification(
        n_samples=1000, 
        n_features=50, 
        n_informative=10,
        n_classes=3, 
        random_state=42
    )
    
    # 2. Data preprocessing
    X_scaled = StandardScaler().fit_transform(X)
    
    # 3. PCA dimensionality reduction (retain 95% variance)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Train linear SVM
    svm = SVC(
        kernel='linear',
        C=1,
        decision_function_shape='ovo',
        class_weight='balanced',
        random_state=42
    )
    svm.fit(X_pca, y)
    
    # 5. Calculate component contributions
    explained_variance = pca.explained_variance_ratio_
    
    # Get SVM weights (multi-class handling)
    if len(svm.coef_.shape) == 2:
        avg_weights = np.mean(np.abs(svm.coef_), axis=0)
    else:
        avg_weights = np.abs(svm.coef_[0])
    
    # Weight contributions by explained variance
    contributions = avg_weights * explained_variance
    
    # 7. Complexity analysis
    # Calculate entropy (measure of contribution distribution)
    contrib_normalized = contributions / np.sum(contributions)
    entropy = -np.sum(contrib_normalized * np.log2(contrib_normalized + 1e-10))
    print(f"Contribution Entropy: {entropy:.3f} bits")
    
    # Calculate Gini index for sparsity
    gini = 1 - np.sum(contrib_normalized**2)
    print(f"Gini Index (0=uniform, 1=concentrated): {gini:.3f}")
    
    # 8. Interpretation guidelines
    print("\nInterpretation guidelines:")
    if entropy > 2.0:
        print("- High entropy indicates classification relies on multiple components")
    else:
        print("- Low entropy suggests classification depends on few components")
    
    if gini > 0.6:
        print("- High Gini index shows contributions are concentrated")
    elif gini < 0.3:
        print("- Low Gini index indicates relatively uniform contributions")
    else:
        print("- Moderate Gini index shows balanced contribution distribution")
        
    # 9. Visualization: Contribution plot
    plt.figure(figsize=(12, 6))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Times New Roman'
    ax1 = plt.gca()
    
    # Contribution bar plot
    bars = ax1.bar(
        range(1, len(contributions) + 1),
        contributions,
        color='skyblue',
        label='Contribution'
    )
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Weighted Contribution', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    
    # Cumulative variance line plot
    ax2 = ax1.twinx()
    cumulative_variance = np.cumsum(explained_variance)
    ax2.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        color='tomato',
        marker='o',
        linestyle='--',
        label='Cumulative Variance'
    )
    ax2.set_ylabel('Cumulative Explained Variance', color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax2.set_ylim(0, 1.1)
    
    # Annotate important components
    max_contri = np.max(contributions)
    for i, (contri, var) in enumerate(zip(contributions, explained_variance)):
        if contri > 0.5 * max_contri:
            ax1.text(i + 1, contri, f'PC{i+1}\n({var*100:.1f}%)',
                    ha='center', va='bottom', fontsize=8)
    
    plt.title(f'PCA Component Contributions to SVM Classification\nEntropy={entropy:.2f}, Gini={gini:.2f}')
    ax1.set_xticks(range(1, len(contributions) + 1))
    plt.tight_layout()
    plt.savefig(f'pca_contributions_{file.split("/")[-1]}.svg', bbox_inches='tight')
    plt.show()
    
    # 10. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 11. Run ablation study (first 20 components)
    max_components = min(20, X_train.shape[1])
    svm_acc, lr_acc = ablation_study(X_train, X_test, y_train, y_test, max_components)

    # 12. Train full models
    svm = SVC(kernel='linear', class_weight='balanced', random_state=42).fit(X_train, y_train)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42).fit(X_train, y_train)

    # 13. Calculate contributions
    svm_contri = get_contributions(svm, pca)
    lr_contri = get_contributions(lr, pca)
    
    # 14. Contribution correlation analysis
    corr, p_value = pearsonr(svm_contri, lr_contri)
    print(f"\nContribution correlation: Pearson r = {corr:.3f} (p = {p_value:.3e})")

    # 15. Comprehensive visualization
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Panel 1: Ablation curves
    ax[0].plot(svm_acc, 'o-', label='SVM')
    ax[0].plot(lr_acc, 's--', label='Logistic Regression')
    ax[0].set_title('Ablation Performance')
    ax[0].axvline(np.argmax(svm_acc) + 1, color='r', linestyle='--', alpha=0.5)
    ax[0].set_xlabel('Number of Components')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    # Panel 2: Contribution correlation scatter plot
    ax[1].scatter(svm_contri, lr_contri, c=range(len(svm_contri)))
    ax[1].plot([0, max(svm_contri)], [0, max(lr_contri)], 'r--')
    ax[1].set_title(f'Contribution Correlation (r = {corr:.3f})')
    ax[1].set_xlabel('SVM Contribution')
    ax[1].set_ylabel('LR Contribution')
    
    # Panel 3: Sorted contribution comparison
    sorted_svm = np.sort(svm_contri)[::-1]
    sorted_lr = np.sort(lr_contri)[::-1]
    ax[2].plot(sorted_svm, 'o-', label='SVM')
    ax[2].plot(sorted_lr, 's--', label='Logistic Regression')
    ax[2].set_title('Sorted Contribution Comparison')
    ax[2].set_xlabel('Rank')
    ax[2].set_ylabel('Contribution')
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'contribution_correlation_{file.split("/")[-1]}.svg', bbox_inches='tight')
    plt.show()