import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
        
# ========= Define function to compute mean squared error (MSE) using KFold ============
def cv_mse(X_data, y_data, alpha, n_splits=10):
    """Compute cross-validated mean squared error for Lasso regression"""
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

def classify(row):
    """Classify neuron response types based on statistical significance"""
    if row['social'] and row['cage']:
        return 'conjunctive-response'
    elif row['social'] and not row['cage']:
        return 'social-response'
    elif not row['social'] and row['cage']:
        return 'corner-response'
    elif not row['social'] and not row['cage'] and row['interaction']:
        return 'combination'
    else:
        return 'non-response'

def cv_metrics(X_data, y_data, alpha, n_splits=10):
    """Compute cross-validated MSE and R² for Lasso regression"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    r2_list = []  # List to store R² scores
    
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred)**2)
        mse_list.append(mse)
        
        r2 = r2_score(y_test, y_pred)  # Calculate R²
        r2_list.append(r2)
    
    return np.array(mse_list), np.array(r2_list)  # Return both arrays

#%% Main processing
# File list placeholder - users should update with their own data paths
file_list = [
    'data/sample_dataset_1',  # Placeholder path 1
    'data/sample_dataset_2'   # Placeholder path 2
]

scaler = StandardScaler()

for day_n, file in enumerate(file_list):
    print(f"Processing: {file}")
    
    file_list_temp = []
    data_len = []
    
    # Load data from two sessions
    for i in range(1, 3):
        # Example file loading - users should adjust to their data structure
        # file_temp = f'{file}_population_decoding_dataset_for_ses{i}.csv'
        # df_temp = pd.read_csv(file_temp)
        
        # For demonstration, create synthetic data
        n_samples = 100
        n_neurons = 50
        synthetic_data = np.random.randn(n_samples, n_neurons)
        columns = [f'neuron_{i}' for i in range(n_neurons)]
        df_temp = pd.DataFrame(synthetic_data, columns=columns)
        
        # Add metadata columns (example structure)
        df_temp['label'] = np.random.choice(['R', 'P', 'N1', 'N2'], n_samples)
        df_temp['cage'] = np.random.choice([1, 2, 3, 4], n_samples)
        df_temp['x'] = np.random.randn(n_samples)
        df_temp['y'] = np.random.randn(n_samples)
        
        file_list_temp.append(df_temp)
        data_len.append(len(df_temp))
    
    # Combine datasets
    decoding_dataset_temp = pd.concat(file_list_temp, axis=0, join='inner', ignore_index=False)
    GLM_prepare = decoding_dataset_temp.iloc[:, 6:]  # Adjust based on actual data structure
    
    # Reset index if needed
    GLM_prepare.reset_index(inplace=True, drop=True)
    
    # Create one-hot encoded features for cage-label combinations
    for label in ['R', 'P', 'N1', 'N2']:
        for cage in [1, 2, 3, 4]:
            mask = (decoding_dataset_temp.cage == cage) & (decoding_dataset_temp.label == label)
            GLM_prepare[f'C{cage}L{label}'] = np.array(mask).astype(int)
    
    # Step 1: Use LassoCV to find optimal alpha
    data = GLM_prepare
    X_full = data.iloc[:, -16:]  # Last 16 columns are the one-hot encoded features
    
    # Filter out columns with no variance (all zeros)
    X_full_filtered = X_full.loc[:, (X_full != 0).any(axis=0)]
    X_full_filtered_array = np.array(X_full_filtered)
    
    units = data.columns[:-16]  # Neural activity columns
    results = []
    
    for neuron in units:
        y = data[neuron].values
        y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
        
        # Find optimal alpha using cross-validation
        lasso_cv = LassoCV(cv=10, random_state=42)
        lasso_cv.fit(X_full_filtered_array, y)
        alpha_best = lasso_cv.alpha_
        print(f"Optimal alpha for {neuron}: {alpha_best:.4f}")
        
        # Step 2: Compute CV metrics for full model
        full_mse, full_r2 = cv_metrics(X_full_filtered_array, y, alpha_best)
        
        # Steps 3-4: Build reduced models for each feature and compare errors
        p_values = []
        for i in range(len(X_full_filtered.columns)):
            # Remove the i-th feature
            X_reduced = np.delete(X_full_filtered_array, i, axis=1)
            reduced_mse = cv_mse(X_reduced, y, alpha_best)
            
            # Paired t-test using errors from 10 folds
            if reduced_mse.mean() > full_mse.mean():
                t_stat, p_val = ttest_rel(reduced_mse, full_mse)
                p_values.append(p_val)
            else:
                p_values.append(1)
        
        # Step 5: Organize results
        p_value_table = pd.DataFrame([p_values], columns=X_full_filtered.columns)
        p_value_table.insert(0, 'R2', [np.mean(full_r2)])
        p_value_table.insert(0, 'neuron', [neuron])
        
        results.append(p_value_table)
    
    # Combine results for all neurons
    result_df = pd.concat(results, ignore_index=True)
    
    # Initialize final result dataframe
    result_df_final = result_df.copy()
    
    # Summarize results by label (social conditions)
    for label in ['R', 'P', 'N1', 'N2']:
        social_to_test = []
        p_to_test = []
        for cage in [1, 2, 3, 4]:
            try:
                social_to_test.append(result_df[f'C{cage}L{label}'] < 0.01)
                p_to_test.append(result_df[f'C{cage}L{label}'])
            except KeyError:
                continue
        
        if social_to_test:  # Check if list is not empty
            social_to_test_df = pd.concat(social_to_test, axis=1)
            p_to_test_df = pd.concat(p_to_test, axis=1)
            result_df_final[label] = social_to_test_df.all(axis=1)
            result_df_final[f'{label}_p'] = p_to_test_df.max(axis=1)
    
    # Summarize results by cage (spatial conditions)
    for cage in [1, 2, 3, 4]:
        cage_to_test = []
        p_to_test = []
        for label in ['R', 'P', 'N1', 'N2']:
            try:
                cage_to_test.append(result_df[f'C{cage}L{label}'] < 0.01)
                p_to_test.append(result_df[f'C{cage}L{label}'])
            except KeyError:
                continue
        
        if cage_to_test:  # Check if list is not empty
            cage_to_test_df = pd.concat(cage_to_test, axis=1)
            p_to_test_df = pd.concat(p_to_test, axis=1)
            result_df_final[str(cage)] = cage_to_test_df.all(axis=1)
            result_df_final[f'{cage}_p'] = p_to_test_df.max(axis=1)
    
    # Calculate summary statistics
    result_df_final['social'] = result_df_final[['R', 'P', 'N1', 'N2']].any(axis=1)
    result_df_final['social_p'] = result_df_final[['R_p', 'P_p', 'N1_p', 'N2_p']].min(axis=1)
    result_df_final['cage'] = result_df_final[['1', '2', '3', '4']].any(axis=1)
    result_df_final['cage_p'] = result_df_final[['1_p', '2_p', '3_p', '4_p']].min(axis=1)
    result_df_final['interaction'] = (result_df.iloc[:, 2:] < 0.01).any(axis=1)
    result_df_final['Type'] = result_df_final.apply(classify, axis=1)
    result_df_final['multi_demo'] = result_df_final[['R', 'P', 'N1', 'N2']].sum(axis=1)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    result_df_final.to_csv(output_dir / f'{Path(file).name}_units_result_R2_v3.csv', index=False)
    
    print(f"Completed processing {file}. Results saved.")

print("All files processed successfully!")