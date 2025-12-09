# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 21:36:41 2025

@author: yuyu2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import chain
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats


def optimize_model(model, param_grid, X_train, y_train):
    """Optimize model hyperparameters using grid search"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.index.name = 'Class'
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fmt = lambda x: "{:.0f}".format(x)
    conf_matrix_fmt = np.vectorize(fmt)(conf_matrix)
    
    plt.figure(figsize=(8, 8))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Times New Roman'
    
    sns.heatmap(conf_matrix, annot=conf_matrix_fmt, fmt='s', cmap='Blues', cbar=False,
                xticklabels=['N1', 'N2', 'P', 'R'], 
                yticklabels=['N1', 'N2', 'P', 'R'], 
                square=True)
    
    plt.title(f'Confusion Matrix (Accuracy: {model.score(X_test, y_test):.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]), 
                    ha='center', va='center', color='brown')
    
    plt.show()
    
    return acc, f1, report_df


def calculate_involved_accuracy(units, unit_this_type, df_decoding, best_rbf, num_involved):
    """Calculate accuracy with a subset of neurons"""
    # Randomly select units
    units_random = np.random.choice(unit_this_type, size=num_involved, replace=False)
    mask_random = ['label'] + list(units_random)
    df_random_situation = df_decoding[mask_random]
    
    # Prepare data
    X = df_random_situation[list(df_random_situation.columns)[1:]]
    X.columns = X.columns.astype(str)
    y = df_random_situation.label
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rus = RandomUnderSampler(sampling_strategy='auto')
    X_train, y_train = rus.fit_resample(X_train, y_train)
    
    # Train model with fixed hyperparameters
    rbf_svm = SVC(kernel='rbf')
    rbf_param_grid = {
        'C': [best_rbf.C],
        'gamma': [best_rbf.gamma]
    }
    best_rbf_local = optimize_model(rbf_svm, rbf_param_grid, X_train, y_train)
    rbf_acc, rbf_f1, report_df = evaluate_model(best_rbf_local, X_test, y_test)
    
    return [rbf_acc, rbf_f1], report_df


def load_and_prepare_data(file_list, condition='pre'):
    """Load and prepare data for analysis"""
    df_N1, df_N2, df_P, df_R = [], [], [], []
    columns_list = []
    
    for file in file_list:
        file_list_temp = []
        for i in range(1, 3):
            # Example file path - users should adjust to their data structure
            file_temp = f'{file}_population_decoding_dataset_for_ses{i}.csv'
            # For demonstration, create synthetic data
            n_samples = 100
            n_neurons = 100
            synthetic_data = np.random.randn(n_samples, n_neurons)
            columns = [f'{file.name}_neuron_{j}' for j in range(n_neurons)]
            df_temp = pd.DataFrame(synthetic_data, columns=columns)
            df_temp['label'] = np.random.choice(['N1', 'N2', 'P', 'R'], n_samples)
            file_list_temp.append(df_temp)
        
        decoding_dataset_temp = pd.concat(file_list_temp, axis=0, join='inner', ignore_index=False)
        decoding_dataset_prefix = decoding_dataset_temp.add_prefix(f'{file.name}_').iloc[:, 6:]
        columns_list.append(decoding_dataset_prefix.columns)
        
        # Separate by condition
        df_N1.append(decoding_dataset_prefix[decoding_dataset_temp.label == 'N1'].reset_index(drop=True))
        df_N2.append(decoding_dataset_prefix[decoding_dataset_temp.label == 'N2'].reset_index(drop=True))
        df_P.append(decoding_dataset_prefix[decoding_dataset_temp.label == 'P'].reset_index(drop=True))
        df_R.append(decoding_dataset_prefix[decoding_dataset_temp.label == 'R'].reset_index(drop=True))
    
    # Combine all columns
    flattened_list = list(chain(*columns_list))
    
    # Create final dataframes for each condition
    df_N1_final = pd.concat(df_N1, axis=1, ignore_index=True)
    df_N1_final.columns = flattened_list
    df_N1_final = df_N1_final.dropna(how='any')
    df_N1_final.insert(0, 'label', 'N1')
    
    df_N2_final = pd.concat(df_N2, axis=1, ignore_index=True)
    df_N2_final.columns = flattened_list
    df_N2_final = df_N2_final.dropna(how='any')
    df_N2_final.insert(0, 'label', 'N2')
    
    df_P_final = pd.concat(df_P, axis=1, ignore_index=True)
    df_P_final.columns = flattened_list
    df_P_final = df_P_final.dropna(how='any')
    df_P_final.insert(0, 'label', 'P')
    
    df_R_final = pd.concat(df_R, axis=1, ignore_index=True)
    df_R_final.columns = flattened_list
    df_R_final = df_R_final.dropna(how='any')
    df_R_final.insert(0, 'label', 'R')
    
    # Combine all conditions
    df_decoding = pd.concat([df_N1_final, df_N2_final, df_P_final, df_R_final], 
                           axis=0, join='inner', ignore_index=False)
    
    return df_decoding


def analyze_feature_curves(units, df_decoding, best_rbf, condition='pre'):
    """Analyze performance curves for different numbers of features"""
    involved_f1 = []
    involved_acc = []
    
    for num_involved in range(5, 260, 5):
        print(f"Testing {num_involved} features")
        
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(calculate_involved_accuracy)(units, units, df_decoding, best_rbf, num_involved)
            for _ in range(50)
        )
        
        results_array = np.array([acc_f1 for acc_f1, _ in results])
        involved_acc.append(results_array[:, 0])
        involved_f1.append(results_array[:, 1])
    
    involved_f1_array = np.array(involved_f1)
    involved_acc_array = np.array(involved_acc)
    
    involved_f1_mean = np.mean(involved_f1_array, axis=1)
    involved_acc_mean = np.mean(involved_acc_array, axis=1)
    involved_f1_std = np.std(involved_f1_array, axis=1)
    involved_acc_std = np.std(involved_acc_array, axis=1)
    
    return involved_f1_mean, involved_f1_std, involved_acc_mean, involved_acc_std


def plot_performance_comparison(pre_f1_mean, pre_f1_std, post_f1_mean, post_f1_std):
    """Plot comparison of pre vs post conditioning performance"""
    x_values = list(range(5, 260, 5))
    
    plt.figure(figsize=(10, 6))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Pre conditioning
    plt.errorbar(x_values, pre_f1_mean, yerr=pre_f1_std, fmt='none', 
                 capsize=5, label='Pre', c='black')
    plt.plot(x_values, pre_f1_mean, marker='o', c='black', markersize=3)
    
    # Post conditioning
    plt.errorbar(x_values, post_f1_mean, yerr=post_f1_std, fmt='none', 
                 capsize=5, label='Post', c='red')
    plt.plot(x_values, post_f1_mean, marker='o', c='red', markersize=3)
    
    plt.xlabel('Number of Neurons')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.ylim([0, 1])
    plt.title('Pre vs. Post Conditioning Performance')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_f1_distribution(f1_scores, real_score, condition_name):
    """Plot distribution of F1 scores for a specific condition"""
    plt.figure(figsize=(8, 6))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.hist(f1_scores, bins=30, color='grey', alpha=0.7, edgecolor='black')
    
    # Calculate p-value
    median_sim = np.median(f1_scores)
    diff_observed = abs(real_score - median_sim)
    diffs_sim = np.abs(f1_scores - median_sim)
    p_value_two_sided = np.mean(diffs_sim >= diff_observed)
    
    plt.axvline(real_score, color='red', linestyle='dashed', 
                linewidth=2, label=f'Real F1 = {real_score:.3f}')
    plt.axvline(np.percentile(f1_scores, 2.5), color='black', 
                linestyle='dotted', linewidth=2, label='95% CI')
    plt.axvline(np.percentile(f1_scores, 97.5), color='black', 
                linestyle='dotted', linewidth=2)
    
    plt.title(f'F1 Score Distribution - {condition_name} (p = {p_value_two_sided:.4f})')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """Main analysis function"""
    # Example paths - users should update with their actual data paths
    home_file = Path('data/analysis')
    
    # Example file lists - users should update with their actual file paths
    pre_files = [
        home_file / 'pre/sample_dataset_1',
        home_file / 'pre/sample_dataset_2',
        home_file / 'pre/sample_dataset_3'
    ]
    
    post_files = [
        home_file / 'post/sample_dataset_1',
        home_file / 'post/sample_dataset_2',
        home_file / 'post/sample_dataset_3'
    ]
    
    # Create results directory
    results_dir = home_file / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze pre-conditioning data
    print("Loading pre-conditioning data...")
    df_decoding_pre = load_and_prepare_data(pre_files, 'pre')
    
    units = list(df_decoding_pre.columns)[1:]
    X = df_decoding_pre[units]
    X.columns = X.columns.astype(str)
    y = df_decoding_pre.label
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    
    # Train initial model with grid search
    print("Training initial model with grid search...")
    rbf_svm = SVC(kernel='rbf')
    rbf_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
    }
    best_rbf_pre = optimize_model(rbf_svm, rbf_param_grid, X_train, y_train)
    
    # Evaluate initial model
    print("Evaluating initial model...")
    rbf_acc_pre, rbf_f1_pre, report_df_pre = evaluate_model(best_rbf_pre, X_test, y_test)
    
    # Analyze feature curves for pre-conditioning
    print("Analyzing feature curves for pre-conditioning...")
    pre_f1_mean, pre_f1_std, pre_acc_mean, pre_acc_std = analyze_feature_curves(
        units, df_decoding_pre, best_rbf_pre, 'pre'
    )
    
    # Analyze post-conditioning data
    print("\nLoading post-conditioning data...")
    df_decoding_post = load_and_prepare_data(post_files, 'post')
    
    units_post = list(df_decoding_post.columns)[1:]
    X_post = df_decoding_post[units_post]
    X_post.columns = X_post.columns.astype(str)
    y_post = df_decoding_post.label
    
    # Split post data
    X_train_post, X_test_post, y_train_post, y_test_post = train_test_split(
        X_post, y_post, test_size=0.2, random_state=42
    )
    X_train_post, y_train_post = rus.fit_resample(X_train_post, y_train_post)
    
    # Train post model
    print("Training post-conditioning model...")
    best_rbf_post = optimize_model(rbf_svm, rbf_param_grid, X_train_post, y_train_post)
    rbf_acc_post, rbf_f1_post, report_df_post = evaluate_model(best_rbf_post, X_test_post, y_test_post)
    
    # Analyze feature curves for post-conditioning
    print("Analyzing feature curves for post-conditioning...")
    post_f1_mean, post_f1_std, post_acc_mean, post_acc_std = analyze_feature_curves(
        units_post, df_decoding_post, best_rbf_post, 'post'
    )
    
    # Plot comparison
    print("\nPlotting performance comparison...")
    plot_performance_comparison(pre_f1_mean, pre_f1_std, post_f1_mean, post_f1_std)
    
    # Statistical comparison
    if pre_f1_mean.shape == post_f1_mean.shape:
        t_statistic, p_value = stats.ttest_rel(pre_f1_mean, post_f1_mean)
        print(f"Paired t-test: t = {t_statistic:.4f}, p = {p_value:.4f}")
    
    # Analyze F1 score distributions for each condition
    print("\nAnalyzing F1 score distributions...")
    
    # For demonstration, create synthetic F1 scores
    np.random.seed(42)
    f1_N1_sim = np.random.normal(0.7, 0.05, 1000)
    f1_N2_sim = np.random.normal(0.65, 0.06, 1000)
    f1_P_sim = np.random.normal(0.75, 0.04, 1000)
    f1_R_sim = np.random.normal(0.8, 0.03, 1000)
    
    # Plot distributions
    plot_f1_distribution(f1_N1_sim, report_df_post['f1-score'].loc['N1'], 'N1')
    plot_f1_distribution(f1_N2_sim, report_df_post['f1-score'].loc['N2'], 'N2')
    plot_f1_distribution(f1_P_sim, report_df_post['f1-score'].loc['P'], 'P')
    plot_f1_distribution(f1_R_sim, report_df_post['f1-score'].loc['R'], 'R')
    
    print("\nAnalysis completed successfully!")
    
    # Save results
    results_summary = {
        'pre_accuracy': rbf_acc_pre,
        'pre_f1': rbf_f1_pre,
        'post_accuracy': rbf_acc_post,
        'post_f1': rbf_f1_post,
        'pre_feature_curve_f1_mean': pre_f1_mean.tolist(),
        'pre_feature_curve_f1_std': pre_f1_std.tolist(),
        'post_feature_curve_f1_mean': post_f1_mean.tolist(),
        'post_feature_curve_f1_std': post_f1_std.tolist()
    }
    
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv(results_dir / 'analysis_results_summary.csv', index=False)
    
    print(f"Results saved to: {results_dir / 'analysis_results_summary.csv'}")


if __name__ == "__main__":
    main()