# -*- coding: utf-8 -*-
"""
Population decoding analysis for social behavior neural data.
This script performs population-level neural decoding using various classifiers
to analyze social interaction representations in neural activity.

Author: yuyu2
Date: Sat Jun  7 09:59:36 2025

Key functions:
1. Load pre-defined neuron populations for different selectivity types
2. Perform population decoding using SVM and Random Forest classifiers
3. Compare linear vs. nonlinear decoding performance
4. Generate performance metrics and visualizations
5. Perform statistical significance testing
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.stats import ttest_rel
import seaborn as sns
from NCP_lty import *
from itertools import chain
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from pathlib import Path

# Load pre-defined neuron populations (example paths - replace with your data)
def load_neuron_populations(base_path='your_data_directory'):
    """Load different neuron populations based on selectivity types."""
    
    # Example neuron populations - replace with your actual data loading
    units_all = pd.read_csv(f'{base_path}/pre_conditioning_units_summary_all.csv').neuron
    units_demo = pd.read_csv(f'{base_path}/pre_conditioning_social_units.csv').neuron
    units_place = pd.read_csv(f'{base_path}/pre_conditioning_spatial_units.csv').neuron
    units_complex = pd.read_csv(f'{base_path}/pre_conditioning_complex_units.csv').neuron
    units_demoplace = pd.read_csv(f'{base_path}/pre_conditioning_spatial+social_units.csv').neuron
    units_nonselective = pd.read_csv(f'{base_path}/pre_conditioning_non-selective_units.csv').neuron
    
    dataset_types = [units_demo, units_place, units_complex, 
                     units_demoplace, units_nonselective]
    type_names = ['social', 'spatial', 'complex', 
                  'social+spatial', 'nonselective']
    
    return dataset_types, type_names

# Population decoding analysis function
def population_decoding_analysis(data_path, save_results=True):
    """
    Perform population decoding analysis on neural data.
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory
    save_results : bool
        Whether to save results to files
    """
    
    # Define data directory structure
    base_path = Path(data_path)
    session_dirs = [
        # Add your session directories here
        # Example: 'path/to/session1', 'path/to/session2'
    ]
    
    for session_dir in session_dirs:
        session_path = Path(session_dir)
        print(f"\nProcessing: {session_path.name}")
        
        # Load population decoding datasets
        decoding_datasets = []
        for i in range(1, 3):  # Assuming 2 sessions
            dataset_file = session_path / f'population_decoding_dataset_for_ses{i}.csv'
            if dataset_file.exists():
                df = pd.read_csv(dataset_file)
                df = df.add_prefix(f'{session_path.name[-15:]}_').iloc[:, 6:]
                df['label'] = df['label'] if 'label' in df.columns else None
                df['cage'] = df['cage'] if 'cage' in df.columns else None
                decoding_datasets.append(df)
            else:
                print(f"Warning: File not found: {dataset_file}")
                continue
        
        if not decoding_datasets:
            print(f"No valid datasets found for {session_path.name}")
            continue
            
        # Combine datasets
        combined_data = pd.concat(decoding_datasets, axis=0, join='inner', ignore_index=False)
        
        # Filter data (remove label 0)
        df_decoding = combined_data[combined_data['label'] != '0']
        
        if df_decoding.empty:
            print(f"No valid data after filtering for {session_path.name}")
            continue
            
        # Prepare features and labels
        units = [col for col in df_decoding.columns if col not in ['label', 'cage']]
        X = df_decoding[units]
        X.columns = X.columns.astype(str)
        y = df_decoding.label
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Handle class imbalance
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        
        # Print dataset info
        print(f"Dataset shape: {X.shape}")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create output directory
        output_dir = session_path.parent / 'individual_decoding'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results file
        results_file = output_dir / f'{session_path.name}_decoding_results.txt'
        
        # Model optimization function
        def optimize_model(model, param_grid, X_train, y_train):
            """Perform grid search for model optimization."""
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
            
            with open(results_file, 'a') as f:
                f.write(f"\nBest parameters: {grid_search.best_params_}")
                f.write(f"Best CV F1 score: {grid_search.best_score_:.4f}\n")
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        
        # Cross-validation scoring function
        def cross_val_scores(model, X, y, n_splits=5):
            """Calculate cross-validation scores."""
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            f1_scores = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)
                y_pred = model_clone.predict(X_test_fold)
                f1 = f1_score(y_test_fold, y_pred, average='macro')
                f1_scores.append(f1)
            
            return np.array(f1_scores)
        
        # Model evaluation function
        def evaluate_model(model, X_test, y_test, model_name):
            """Evaluate model performance and generate metrics."""
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            print(f"{model_name} - Test accuracy: {acc:.4f}")
            print(f"{model_name} - Test F1 score: {f1:.4f}")
            print("\nClassification report:")
            print(classification_report(y_test, y_pred))
            
            with open(results_file, 'a') as f:
                f.write(f"\n{model_name} Performance:")
                f.write(f"Test accuracy: {acc:.4f}")
                f.write(f"Test F1 score: {f1:.4f}")
                f.write("\nClassification report:\n")
                f.write(classification_report(y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(y),
                        yticklabels=np.unique(y))
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_results:
                plt.savefig(output_dir / f'{session_path.name}_{model_name}_confusion_matrix.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
            return acc, f1
        
        # Start analysis
        with open(results_file, 'w') as f:
            f.write(f"Population Decoding Analysis Results\n")
            f.write(f"Session: {session_path.name}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n")
            f.write("=" * 60 + "\n")
        
        # 1. Linear SVM
        print("\n" + "="*50)
        print("Optimizing Linear SVM...")
        with open(results_file, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write("Linear SVM Optimization\n")
        
        linear_svm = SVC(kernel='linear', random_state=42)
        linear_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        best_linear = optimize_model(linear_svm, linear_param_grid, X_train, y_train)
        linear_acc, linear_f1 = evaluate_model(best_linear, X_test, y_test, "Linear_SVM")
        
        # 2. RBF Kernel SVM
        print("\n" + "="*50)
        print("Optimizing RBF Kernel SVM...")
        with open(results_file, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write("RBF Kernel SVM Optimization\n")
        
        rbf_svm = SVC(kernel='rbf', random_state=42)
        rbf_param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
        }
        best_rbf = optimize_model(rbf_svm, rbf_param_grid, X_train, y_train)
        rbf_acc, rbf_f1 = evaluate_model(best_rbf, X_test, y_test, "RBF_SVM")
        
        # 3. Random Forest
        print("\n" + "="*50)
        print("Optimizing Random Forest...")
        with open(results_file, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write("Random Forest Optimization\n")
        
        rf = RandomForestClassifier(random_state=42)
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        best_rf = optimize_model(rf, rf_param_grid, X_train, y_train)
        rf_acc, rf_f1 = evaluate_model(best_rf, X_test, y_test, "Random_Forest")
        
        # Statistical significance testing
        print("\n" + "="*50)
        print("Statistical Significance Testing...")
        with open(results_file, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write("Statistical Significance Tests\n")
        
        # Get cross-validation scores
        linear_scores = cross_val_scores(best_linear, np.array(X_train), np.array(y_train))
        rbf_scores = cross_val_scores(best_rbf, np.array(X_train), np.array(y_train))
        
        # Paired t-test
        t_stat, p_value = ttest_rel(linear_scores, rbf_scores)
        print(f"Linear SVM vs RBF SVM: t = {t_stat:.4f}, p = {p_value:.4f}")
        
        with open(results_file, 'a') as f:
            f.write(f"\nLinear SVM vs RBF SVM comparison:\n")
            f.write(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}\n")
            
            alpha = 0.05
            if p_value < alpha:
                if np.mean(rbf_scores) > np.mean(linear_scores):
                    f.write("RBF SVM significantly outperforms Linear SVM (p < 0.05)\n")
                    print("RBF SVM significantly outperforms Linear SVM")
                else:
                    f.write("Linear SVM significantly outperforms RBF SVM (p < 0.05)\n")
                    print("Linear SVM significantly outperforms RBF SVM")
            else:
                f.write("No significant difference between Linear and RBF SVM (p >= 0.05)\n")
                print("No significant difference between Linear and RBF SVM")
        
        # Performance comparison plot
        models = ['Linear SVM', 'RBF SVM', 'Random Forest']
        acc_scores = [linear_acc, rbf_acc, rf_acc]
        f1_scores = [linear_f1, rbf_f1, rf_f1]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, acc_scores, width, label='Accuracy')
        rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')
        
        ax.set_ylabel('Score')
        ax.set_title(f'Model Performance Comparison - {session_path.name}')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        
        if save_results:
            plt.savefig(output_dir / f'{session_path.name}_model_comparison.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Final conclusions
        print("\n" + "="*50)
        print("Analysis Conclusions:")
        print(f"Linear SVM test F1 score: {linear_f1:.4f}")
        print(f"RBF SVM test F1 score: {rbf_f1:.4f}")
        print(f"Random Forest test F1 score: {rf_f1:.4f}")
        
        with open(results_file, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write("Analysis Conclusions:\n")
            f.write(f"Linear SVM test F1 score: {linear_f1:.4f}\n")
            f.write(f"RBF SVM test F1 score: {rbf_f1:.4f}\n")
            f.write(f"Random Forest test F1 score: {rf_f1:.4f}\n")
            
            if rbf_f1 - linear_f1 > 0.05 and p_value < 0.05:
                f.write("\nConclusion: Significant nonlinear structure present in data\n")
                f.write("Recommendation: Use nonlinear models (RBF SVM or Random Forest)\n")
                print("\nConclusion: Significant nonlinear structure present in data")
                print("Recommendation: Use nonlinear models (RBF SVM or Random Forest)")
            elif linear_f1 > rbf_f1 - 0.02 and p_value >= 0.05:
                f.write("\nConclusion: Linear models sufficiently represent the data\n")
                f.write("Recommendation: Use linear SVM for simplicity and interpretability\n")
                print("\nConclusion: Linear models sufficiently represent the data")
                print("Recommendation: Use linear SVM for simplicity and interpretability")
            else:
                f.write("\nConclusion: Mild nonlinear structure may be present\n")
                f.write("Recommendation: Choose model based on specific needs\n")
                print("\nConclusion: Mild nonlinear structure may be present")
                print("Recommendation: Choose model based on specific needs")
        
        print(f"\nAnalysis complete for {session_path.name}")
        print(f"Results saved to: {output_dir}")

# Main execution block
if __name__ == "__main__":
    # Example usage
    data_directory = "path/to/your/data/directory"
    
    # Load neuron populations
    dataset_types, type_names = load_neuron_populations(data_directory)
    
    # Run decoding analysis for each dataset type
    for dataset, name in zip(dataset_types, type_names):
        print(f"\n{'='*60}")
        print(f"Analyzing {name} neurons")
        print(f"{'='*60}")
        
        # Here you would filter your data based on the neuron IDs in 'dataset'
        # and then run the decoding analysis
        # For demonstration, we'll just print the number of neurons
        print(f"Number of {name} neurons: {len(dataset)}")
    
    # Run population decoding analysis
    # Replace 'example_session_path' with your actual data path
    population_decoding_analysis(
        data_path="example_session_path",
        save_results=True
    )