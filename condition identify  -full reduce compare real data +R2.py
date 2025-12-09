# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 09:18:38 2025

@author: yuyu2
"""
from NCP_lty import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV,ElasticNet
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit  

def align_columns(df, target_columns, fill_value=0):
    """Add missing columns to a DataFrame and fill them with specified values."""
    missing_cols = set(target_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = fill_value
    return df[target_columns]

def build_delay_features(stim_series, max_delay=300, bin_size=250):  
    """  
    stim_series: Stimulus time series (1 indicates stimulus onset)
    max_delay: Maximum delay time (ms)
    bin_size: Time bin size (ms) 
    Returns: A delayed feature matrix of shape (n_bins, n_delays)
    """  
    n_delays = int(max_delay / bin_size)  
    X_delay = np.zeros((len(stim_series), n_delays))  
    for i in range(n_delays):  
        X_delay[i+1:, i] = stim_series[:-i-1]  
    return X_delay  
def cv_mse(X_data, y_data, alpha, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        model = ElasticNet(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred)**2)
        mse_list.append(mse)
    return np.array(mse_list)
def cv_metrics(X_data, y_data, alpha, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    r2_list = []  # store R²
    
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        model = ElasticNet(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred)**2)
        mse_list.append(mse)
        
        r2 = r2_score(y_test, y_pred)  # 计算R²
        r2_list.append(r2)
    
    return np.array(mse_list), np.array(r2_list)  # 返回两个数组
def classify(row):
    if row['R sig'] and row['signalR']:
        return 'combination'
    elif row['P sig'] and row['signalP']:
        return 'combination'
    elif row['R sig'] and not row['signalR']:
        return 'social-only-response'
    elif row['P sig'] and not row['signalP']:
        return 'social-only-response'
    elif not row['R sig'] and row['signalR']:
        return 'condition-only-response'
    elif not row['P sig'] and row['signalP']:
        return 'condition-only-response'
    elif row['N1 sig'] or row['N2 sig']:
        return 'social-only-response'
    else:
        return 'no-responce'


file_list=[
        '..\dataincondition\kedou1-20240220', #example, folder_path\mouse&day
   ]


scaler = StandardScaler()
for day_n,file in enumerate(file_list):
    file_list_temp=[]
    data_len=[]
    for i in range(0,4):
        file_temp=str(f'{file}population_decoding_condition_dataset for ses{i}.csv')
        df_temp=pd.read_csv(file_temp)
        file_list_temp.append(df_temp)
        data_len.append(len(df_temp))
    all_columns = list(set().union(*[set(df.columns) for df in file_list_temp]))
    aligned_dfs = [align_columns(df, all_columns) for df in file_list_temp]
    decoding_dataset_temp=pd.concat(file_list_temp,axis=0, join='inner', ignore_index=False)
    GLM_prepare=decoding_dataset_temp.iloc[:,6:-2]
    
    for label in ['R','P','N1','N2']:
        mask=decoding_dataset_temp.label==label
        GLM_prepare[label]=np.array(mask).astype(int)
    for signal in ['signalR','signalP']:
        mask=decoding_dataset_temp[signal]==1
        delay_features=build_delay_features(mask, max_delay=1000, bin_size=250)
        new_columns=[f'{signal}{i}' for i in range(delay_features.shape[1])]
        GLM_prepare[new_columns] = delay_features
        
    #%%
    data=GLM_prepare
    X_full=data.iloc[:,-12:]
    X_full_filtered_array=np.array(X_full)
    units=data.columns[:-12]
    result=[]
    
    for neuron in units:
        y = data[neuron].values
        y= scaler.fit_transform(y.reshape(-1, 1))
        y=y.reshape(-1)
        Elastic_cv = ElasticNetCV(cv=10, random_state=42)
        Elastic_cv.fit(X_full_filtered_array, y)
        alpha_best = Elastic_cv.alpha_
        print("best alpha:", alpha_best)

        full_mse, full_r2 = cv_metrics(X_full_filtered_array, y, alpha_best)
        
        # Reduced model 
        p_values = []
        for i in range(len(X_full.columns)):
            # 删除第 i 个刺激
            X_reduced = np.delete(X_full_filtered_array, i, axis=1)
            reduced_mse = cv_mse(X_reduced, y, alpha_best)  # 同样使用最佳 alpha
            # 利用 10 个折的误差进行配对 t 检验
            if np.mean(reduced_mse)>np.mean(full_mse):
                t_stat, p_val = ttest_rel(reduced_mse, full_mse)
                p_values.append(p_val)
            else:
                p_values.append(1)
        
        #  reslut 
        p_value_table = pd.DataFrame([p_values],
                                     columns=X_full.columns)
        p_value_table.insert(0,'R2',[np.mean(full_r2)])
        p_value_table.insert(0,'neuron',[neuron])
        # p_value_table.insert(0,'neuron',[neuron])
        result.append(p_value_table)
    result_df=pd.concat(result)
    result_df.reset_index()
    result_df_final=result_df.copy()
    
    
    for label in ['R','P','N1','N2']:
        result_df_final[f'{label} sig']=(result_df[label]<0.01)
        
    result_df_final['signalR']=(result_df[['signalR0','signalR1','signalR2','signalR3']]<0.01).any(axis=1)
    result_df_final['signalP']=(result_df[['signalP0','signalP1','signalP2','signalP3']]<0.01).any(axis=1)
    result_df_final['Type'] = result_df_final.apply(classify, axis=1)
    # result_df_final.to_csv(savepath)
    
    
    
    