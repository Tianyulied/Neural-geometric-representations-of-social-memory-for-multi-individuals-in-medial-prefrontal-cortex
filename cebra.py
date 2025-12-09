# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:50:56 2024

@author: yuyu2
"""



#%%
'''
multi-session training
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import pandas as pd
import cebra
from cebra import CEBRA
import tempfile
import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns

def hausdorff_distance(u, v):
    # 计算所有点对之间的距离矩阵
    d = cdist(u, v, 'euclidean')
    
    # 计算从u到v的Hausdorff距离
    max_min_1 = np.max(np.min(d, axis=1))
    
    # 计算从v到u的Hausdorff距离
    max_min_2 = np.max(np.min(d, axis=0))
    
    # 取两者中的最大值作为Hausdorff距离
    return max(max_min_1, max_min_2)

def adaptive_hausdorff_similarity(arrays):
    """
    自适应计算Hausdorff相似度，考虑数据的特性
    
    参数:
    arrays: 包含多个形状为(n, 3)数组的列表
    
    返回:
    相似度矩阵（0-1之间，1表示最相似）
    """
    n_arrays = len(arrays)
    
    # 首先计算所有Hausdorff距离
    distance_matrix = np.zeros((n_arrays, n_arrays))
    for i in range(n_arrays):
        for j in range(i+1, n_arrays):
            dist = hausdorff_distance(arrays[i], arrays[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # 计算数据的整体范围，用于归一化
    all_points = np.vstack(arrays)
    data_range = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    # 使用欧几里得范数作为整体范围度量
    overall_range = np.linalg.norm(data_range)
    
    # 将距离归一化并转换为相似度
    similarity_matrix = np.ones((n_arrays, n_arrays))
    for i in range(n_arrays):
        for j in range(n_arrays):
            if i != j:
                # 归一化距离
                normalized_dist = distance_matrix[i, j] / overall_range
                # 转换为相似度
                similarity_matrix[i, j] = 1 - min(1, normalized_dist)
    
    return similarity_matrix
#%%
datapath=Path('') # the preprocessed data file path
mouse_name_and_day1='kedou1-20240215' #for example

datasets_path =[f'{datapath}/{mouse_name_and_day1}',
             f'{datapath}/{mouse_name_and_day1}',
             f'{datapath}/{mouse_name_and_day1}',
             f'{datapath}/{mouse_name_and_day1}' ]
        
   






datas=[]
labels=[]
dataset = {}

labels=[]
models, embeddings = {}, {}
for i, dataset_path in enumerate(datasets_path):
    ses1=pd.read_csv(f'{dataset_path}population_decoding_dataset for ses1.csv')
    ses2=pd.read_csv(f'{dataset_path}population_decoding_dataset for ses2.csv')

    dataset[f'{Path(dataset_path).name}']=pd.concat([ses1,ses2],axis=0, join='inner', ignore_index=False)
    
    data_temp=dataset[f'{Path(dataset_path).name}'].iloc[:,6:]
    data_temp = np.nan_to_num(data_temp, nan=0.0)
    datas.append(data_temp)
    target= dataset[f'{Path(dataset_path).name}'].label
    target[target=='0']=0
    target[target=='R']=1
    target[target=='P']=2
    target[target=='N1']=3
    target[target=='N2']=4


    label_temp=np.array(target).astype(float)
    labels.append(label_temp)

multi_cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=1024,
                        learning_rate=3e-4,
                        # temperature=1,
                        temperature_mode="auto",
                        output_dimension=3,
                        max_iterations=10000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        # hybrid=True,
                        verbose=True,
                        time_offsets=10)
multi_cebra_model.fit(datas, labels)
multi_embeddings = dict()
for i,  (X,y) in enumerate(zip(datas,labels)):
    embedding_temp = multi_cebra_model.transform(X, session_id=i)
    multi_embeddings[i] = embedding_temp
    cebra.plot_embedding(
                              embedding=embedding_temp, 
                               # embedding_labels=y[:,0], 
                               embedding_labels=y, 
                              title='CEBRA-Behavior', 
                              markersize=0.5,
                              cmap=ListedColormap(['gray','red','brown','blue','green'])
                              )
    plt.axis('off')
    plt.show()

for label2compare in range(5):
    embedings2compare=[]
    for ii,  label in enumerate(labels):
        embeding = multi_embeddings[ii]
        embedings2compare.append(embeding[np.where(label==label2compare)[0],:])
   

    similarity_matrix_3 = adaptive_hausdorff_similarity(embedings2compare)
    
    plt.figure(figsize=(8,6))
    plt.rcParams['svg.fonttype']='none'
    plt.rcParams['font.family']='Times New Roman'
    plt.title(f'{Path(datasets_path[i]).name} demo{label2compare} hausdorff similarity multi demo traning')
    sns.heatmap(similarity_matrix_3, 
                fmt=".2f", 
                cmap="jet", vmin=0, vmax=1,
                xticklabels=[f'day{i+1}' for i in range(similarity_matrix_3.shape[0])],
                yticklabels=[f'day{i+1}' for i in range(similarity_matrix_3.shape[0])],
                )
    for a in range(len(similarity_matrix_3)):
        for b in range(len(similarity_matrix_3)):
            plt.text(b + 0.5, a + 0.5,  f'{np.round(similarity_matrix_3[a, b],2)}', ha='center', va='center', color='black')
    
    # plt.savefig(Path(dataset_path).parent/f'{Path(datasets_path[i]).name} demo{label2compare} hausdorff similarity multi demo traning.svg')
    plt.tight_layout()
    plt.show()
    
for ii,  label in enumerate(labels):
    embedings2compare=[]
    for label2compare in range(5):
        embeding = multi_embeddings[ii]
        embedings2compare.append(embeding[np.where(label==label2compare)[0],:])
   

    similarity_matrix_3 = adaptive_hausdorff_similarity(embedings2compare)

    plt.figure(figsize=(8,6))
    plt.rcParams['svg.fonttype']='none'
    plt.rcParams['font.family']='Times New Roman'
    plt.title(f'{Path(datasets_path[i]).name} day{ii} hausdorff similarity multi demo traning')
    sns.heatmap(similarity_matrix_3, 
                fmt=".2f", 
                cmap="jet", vmin=0, vmax=1,
                xticklabels=['0','R','A','N1','N2'],
                yticklabels=['0','R','A','N1','N2'],
                )
    for a in range(len(similarity_matrix_3)):
        for b in range(len(similarity_matrix_3)):
            plt.text(b + 0.5, a + 0.5,  f'{np.round(similarity_matrix_3[a, b],2)}', ha='center', va='center', color='black')
    
    # plt.savefig(Path(dataset_path).parent/f'{Path(datasets_path[i]).name} day{ii} hausdorff similarity multi demo traning.svg')
    plt.tight_layout()
    plt.show()

    
