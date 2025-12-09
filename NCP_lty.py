# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:50:28 2023

Welcome to NCP lab! The lasy authar so far has not written any informative docstring yet.

@author: Junhao
"""
'''
UPDATES

20230708, for 1d info & single neuron shuffle test, Skaggs info & Olypher info.

20230715. Core reconstructed, by reconstruction of time defined by external clock(Master8).
Which means all V_frames or E_timestamps are translated into time by count of sync pulse,
in other words, full trust was given to Maser8.
Pros, reconstruted time by many interpolations(Scipy, UniviariateSpline) might be easy to use.
Cons, we cannot know if Master8 is really good enough so time-related calculation might be wrong.

20231119 merged codes from Mingze.



'''

#%% Main Bundle


# ----------------------------------------------------------------------------
#    LOTS OF THINGS TO BE DONE.
# ----------------------------------------------------------------------------


# later, mind if there are some nans in DLC files.
# jumpy detection, or smooth, Kalman Filter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# everything is function of time. How to verify the result after spline-interp of xy?? make a new video?



# waveform putative IN or PCs. Then optotag, R of waveforms of units.
# in class unit, furthur work with its quality check like L-ratio and others. May need to load more files from KS&phy2.


# Master8 got a bit faster or DAQ slower? e_intervals are mostly 14998 and none greater than 15000. Errors are accumulating!!!!!!!!!!!!!!!!!!!

# better session, coorperate with behavior-tags 2 frames. 

# LFP&spike, their binding do not need anything related to videos. Well except for spd thresh, or maybe some relation with its position.
# func & methods for 2D exp. , smoothing kernels. More and More
# decoding. some bayesian?



# ----------------------------------------------------------------------------
#                  Packages 
# ----------------------------------------------------------------------------

import brpylib, time, random, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt, numpy_groupies as npg
from scipy import optimize, ndimage, interpolate, stats,signal
from matplotlib.colors import ListedColormap
from pathlib import Path
from tkinter import filedialog
import seaborn, csv
import os
from scipy.stats import binned_statistic,spearmanr
import itertools
import pickle
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy.ndimage.filters import gaussian_filter
from sklearn.utils import resample
from scipy.signal import find_peaks

# ----------------------------------------------------------------------------
#                  Functions here
# ----------------------------------------------------------------------------
def generate_population_decoding_dataset(unit_temp,time2xy_interp, total_time,temporal_bin_length, cage_pos2,cage_line,dist=12):
    t = np.linspace(0, total_time, num=int(total_time/temporal_bin_length), endpoint=False) + 0.5*temporal_bin_length#assign spatial values at midpoint of every time bin.
    pol_temporal_bin = np.array([t,(time2xy_interp[0](t)),(time2xy_interp[1](t))]).T
    pol_temporal_bin_df=pd.DataFrame(pol_temporal_bin,columns=['timepoint','x','y'])
    pol_temporal_bin_df['x'][pol_temporal_bin_df['x']>50]=50
    pol_temporal_bin_df['x'][pol_temporal_bin_df['x']<0]=0
    pol_temporal_bin_df['y'][pol_temporal_bin_df['y']>50]=50
    pol_temporal_bin_df['y'][pol_temporal_bin_df['y']<0]=0
    def distance(x1, y1, x2, y2):
      # 计算两个点之间的欧几里得距离
      return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def label(row):
      # 根据距离判断标签
      x = row['x']
      y = row['y']
      if distance(x, y, cage_pos2['cage1','x'],cage_pos2['cage1','y']) < dist:
        return cage_line[0]
      elif distance(x, y, cage_pos2['cage2','x'],cage_pos2['cage2','y']) < dist:
        return cage_line[1]
      elif distance(x, y, cage_pos2['cage3','x'],cage_pos2['cage3','y']) < dist:
        return cage_line[2]
      elif distance(x, y, cage_pos2['cage4','x'],cage_pos2['cage4','y']) < dist:
        return cage_line[3]
      else:
        return 0
    def label_cage(row):
      # 根据距离判断标签
      x = row['x']
      y = row['y']
      if distance(x, y, cage_pos2['cage1','x'],cage_pos2['cage1','y']) < dist:
        return '1'
      elif distance(x, y, cage_pos2['cage2','x'],cage_pos2['cage2','y']) < dist:
        return '2'
      elif distance(x, y, cage_pos2['cage3','x'],cage_pos2['cage3','y']) < dist:
        return '3'
      elif distance(x, y, cage_pos2['cage4','x'],cage_pos2['cage4','y']) < dist:
        return '4'
      else:
        return '0'
    def mark(df):
      # 对DataFrame的每一行应用label函数，得到一个新的Series对象
      s = df.apply(label, axis=1)
      # 将新的Series对象作为新的一列添加到DataFrame中，命名为label
      df['label'] = s
      # 返回修改后的DataFrame
      s = df.apply(label_cage, axis=1)
      df['cage'] = s
      return df
    for unit in set(unit_temp['cluid']):
        try:
            hist, bins, _ = binned_statistic(unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                             unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                            statistic='count',\
                                                bins=np.arange(0, total_time+0.5*temporal_bin_length, temporal_bin_length))
            pol_temporal_bin_df.insert(pol_temporal_bin_df.shape[1], f'{unit}', hist)
        except:
            hist, bins, _ = binned_statistic(unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                             unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                            statistic='count',\
                                                bins=np.arange(0, total_time, temporal_bin_length))
            pol_temporal_bin_df.insert(pol_temporal_bin_df.shape[1], f'{unit}', hist)
    return pol_temporal_bin_df
    
def generate_population_decoding_condition_dataset(unit_temp,time2xy_interp, total_time, signal_time, temporal_bin_length, cage_pos2,cage_line,tag,dist=12):
    t = np.linspace(0, total_time, num=int(total_time/temporal_bin_length), endpoint=False) + 0.5*temporal_bin_length#assign spatial values at midpoint of every time bin.
    pol_temporal_bin = np.array([t,(time2xy_interp[0](t)),(time2xy_interp[1](t))]).T
    pol_temporal_bin_df=pd.DataFrame(pol_temporal_bin,columns=['timepoint','x','y'])
    pol_temporal_bin_df['x'][pol_temporal_bin_df['x']>50]=50
    pol_temporal_bin_df['x'][pol_temporal_bin_df['x']<0]=0
    pol_temporal_bin_df['y'][pol_temporal_bin_df['y']>50]=50
    pol_temporal_bin_df['y'][pol_temporal_bin_df['y']<0]=0
    def distance(x1, y1, x2, y2):
      # 计算两个点之间的欧几里得距离
      return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def label(row):
      # 根据距离判断标签
      x = row['x']
      y = row['y']
      if distance(x, y, cage_pos2['cage1','x'],cage_pos2['cage1','y']) < dist:
        return cage_line[0]
      elif distance(x, y, cage_pos2['cage2','x'],cage_pos2['cage2','y']) < dist:
        return cage_line[1]
      elif distance(x, y, cage_pos2['cage3','x'],cage_pos2['cage3','y']) < dist:
        return cage_line[2]
      elif distance(x, y, cage_pos2['cage4','x'],cage_pos2['cage4','y']) < dist:
        return cage_line[3]
      else:
        return 0
    def label_cage(row):
      # 根据距离判断标签
      x = row['x']
      y = row['y']
      if distance(x, y, cage_pos2['cage1','x'],cage_pos2['cage1','y']) < dist:
        return '1'
      elif distance(x, y, cage_pos2['cage2','x'],cage_pos2['cage2','y']) < dist:
        return '2'
      elif distance(x, y, cage_pos2['cage3','x'],cage_pos2['cage3','y']) < dist:
        return '3'
      elif distance(x, y, cage_pos2['cage4','x'],cage_pos2['cage4','y']) < dist:
        return '4'
      else:
        return '0'
    def mark(df):
      # 对DataFrame的每一行应用label函数，得到一个新的Series对象
      s = df.apply(label, axis=1)
      # 将新的Series对象作为新的一列添加到DataFrame中，命名为label
      df['label'] = s
      # 返回修改后的DataFrame
      s = df.apply(label_cage, axis=1)
      df['cage'] = s
      return df
    pol_temporal_bin_df=mark(pol_temporal_bin_df)
    if len(signal_time)>0:
        try:
            hist, bins, _ = binned_statistic(signal_time,\
                                             signal_time,\
                                            statistic='count',\
                                                bins=np.arange(0, total_time+0.5*temporal_bin_length, temporal_bin_length))
            pol_temporal_bin_df.insert(pol_temporal_bin_df.shape[1],f'signal{tag}', hist)
        except:
            hist, bins, _ = binned_statistic(signal_time,\
                                             signal_time,\
                                            statistic='count',\
                                                bins=np.arange(0, total_time, temporal_bin_length))
            pol_temporal_bin_df.insert(pol_temporal_bin_df.shape[1],f'signal{tag}', hist)
    for unit in set(unit_temp['cluid']):
        try:
            hist, bins, _ = binned_statistic(unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                             unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                            statistic='count',\
                                                bins=np.arange(0, total_time+0.5*temporal_bin_length, temporal_bin_length))
            pol_temporal_bin_df.insert(pol_temporal_bin_df.shape[1], f'{unit}', hist)
        except:
            hist, bins, _ = binned_statistic(unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                             unit_temp['timepoint'][unit_temp['cluid']==unit],\
                                            statistic='count',\
                                                bins=np.arange(0, total_time, temporal_bin_length))
            pol_temporal_bin_df.insert(pol_temporal_bin_df.shape[1], f'{unit}', hist)
    return pol_temporal_bin_df
    
    

def positional_information_olypher_1dcircular(spiketime, time2xy_interp, total_time, temporal_bin_length, cage_pos2,cage_line,dist=12):
    t = np.linspace(0, total_time, num=int(total_time/temporal_bin_length), endpoint=False) + 0.5*temporal_bin_length#assign spatial values at midpoint of every time bin.
    pol_temporal_bin = np.array([(time2xy_interp[0](t)),(time2xy_interp[1](t))]).T
    pol_temporal_bin[pol_temporal_bin>50]=50
    pol_temporal_bin[pol_temporal_bin<0]=0
    pol_temporal_bin_df=pd.DataFrame(pol_temporal_bin,columns=['x','y'])
    spk_time_temp = (spiketime/temporal_bin_length).astype('uint')
    
    def distance(x1, y1, x2, y2):
      # 计算两个点之间的欧几里得距离
      return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def label(row):
      # 根据距离判断标签
      x = row['x']
      y = row['y']
      if distance(x, y, cage_pos2['cage1','x'],cage_pos2['cage1','y']) < dist:
        return cage_line[0]
      elif distance(x, y, cage_pos2['cage2','x'],cage_pos2['cage2','y']) < dist:
        return cage_line[1]
      elif distance(x, y, cage_pos2['cage3','x'],cage_pos2['cage3','y']) < dist:
        return cage_line[2]
      elif distance(x, y, cage_pos2['cage4','x'],cage_pos2['cage4','y']) < dist:
        return cage_line[3]
      else:
        return 0
    def mark(df):
      # 对DataFrame的每一行应用label函数，得到一个新的Series对象
      s = df.apply(label, axis=1)
      # 将新的Series对象作为新的一列添加到DataFrame中，命名为label
      df['label'] = s
      # 返回修改后的DataFrame
      return df
    pol_temporal_bin_df=mark(pol_temporal_bin_df)
    
    spk_count_temporal_bin = npg.aggregate(spk_time_temp, 1, size=int(total_time/temporal_bin_length)+1)
    p_k = npg.aggregate(spk_count_temporal_bin,1)/np.sum(npg.aggregate(spk_count_temporal_bin,1))
    
    nspatial_bins=5
    pos_info_dict = {}
    # for i in range(nspatial_bins):#好好设计一下如何改这个bins，i和j遍历整个pol_temporal_bin应该就行了
    for i in cage_line:
        spk_count_temporal_bin_xi = spk_count_temporal_bin[np.where(pol_temporal_bin_df['label']==i)]
        if spk_count_temporal_bin_xi.size==0:
            pos_info_dict[f'zone{i}']=0
        else: 
            p_kxi = npg.aggregate(spk_count_temporal_bin_xi, 1)/np.sum(npg.aggregate(spk_count_temporal_bin_xi, 1))
            pos_info_dict[f'zone{i}']=np.nansum(p_kxi * np.log2(p_kxi/p_k[:np.size(p_kxi)]))# set range for p_k is that, e.g. some time bin might have 8 spks or more but not in certain spatial bin, then arrays are not the same length.            # set range for p_k is that, e.g. some time bin might have 8 spks or more but not in certain spatial bin, then arrays are not the same length. 
    # return  pos_info_dict,pol_temporal_bin_df
    return  pos_info_dict
def cohen_d(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    s1 = np.std(x1, ddof=1)
    s2 = np.std(x2, ddof=1)
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    s = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    d = (x1_mean - x2_mean) / s
    return d

def load_files(fdir, fn, Nses, experiment_tag, dlc_tail, mode):  
    # if Nses > 1, mind the rule of name.
    spike_times = np.load(fdir/'spike_times.npy')
    spike_times = np.squeeze(spike_times)# delete that stupid dimension.
    spike_clusters = np.load(fdir/'spike_clusters.npy')
    clusters_quality = pd.read_csv(fdir/'cluster_group.tsv', sep='\t')
    esync_timestamps_load = np.load(fdir.parent/'Esync_timestamps.npy')  

    signal_on_timestamps_load = np.load(fdir.parent/'Signal_on_timestamps.npy')
    
    waveforms = {'raw':0, 'spike_id':0, 'cluster':0, 'timestamp':0, 'channel':0, 'tetrode':0}
    waveforms['raw'] = np.load(fdir/'_phy_spikes_subset.waveforms.npy')[:,11:71,:]  # (randomized select?) n waveforms, 60 for sample length, 16 for 16 channels
    waveforms['spike_id'] = np.load(fdir/'_phy_spikes_subset.spikes.npy')          # waveform belong to which spike  (in order, 1st spike 2nd spike 3rd spike...)
    waveforms['cluster'] = spike_clusters[waveforms['spike_id']]                        # this spike belong to which cluster
    waveforms['timestamp'] = spike_times[waveforms['spike_id']]                         # when happen
    waveforms['channel'] = np.load(fdir/'_phy_spikes_subset.channels.npy')      # waveform happen on which channel, though we use tetrode, only 4 channel contain the useful information, but phy out put continuous 12 channel info
    # waveforms['tetrode'] = waveforms['channel'][:,0]//4                                 # first in channel must be the most significant channel which used to calculate the id of tetrode
    
                
    timestamps = []
    spike_clusters2 = []
    dlc_files = []
    vsync = []
    frame_state = []
    for n in range(1,Nses+1):
        dlc_temp = pd.read_hdf(fdir/(f'dlc_concatenate{n}.h5'))
        dlc_files.append(dlc_temp[dlc_tail[:-12]])
        # '''
        # framestate需要好好去问一下
        # '''
        if mode == 'FrameState':
            frame_state = pd.read_csv(fdir/(f'vsync{n}.csv'))    #Try using this rule to name files.
            vsync_temp = np.array(frame_state['blockID'], dtype='uint')
            vsync.append(vsync_temp)
        elif mode == 'Bonsai':
            vsync_csv = pd.read_csv(fdir/(f'vsync{n}.csv'), names=[0,1,2])
            vsync_temp = np.array(vsync_csv.loc[:,1], dtype='uint')
            vsync.append(np.where((vsync_temp[1:] - vsync_temp[:-1]) ==1)[0])
   
    # arbituarily more than 10s interval would be made when concatenate ephys files.
    
    ses_e_end = esync_timestamps_load[np.where((esync_timestamps_load[1:] - esync_timestamps_load[:-1]) > 100000)[0]]
    ses_e_end = np.append(ses_e_end, esync_timestamps_load[-1])# last one sync needed here.
    esync_timestamps = [esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[0] + 100000)]]
    
    for i in range(1, Nses):
        esync_temp = esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[i] + 100000)]
        esync_temp = esync_temp[np.where(esync_temp > ses_e_end[i-1])]
        esync_timestamps.append(esync_temp)
        
    
    signal_on_timestamps = []
    for i in range(Nses):
        signal_on_temp = signal_on_timestamps_load[np.where(signal_on_timestamps_load < esync_timestamps[i][-1])]
        signal_on_temp = signal_on_temp[np.where(signal_on_temp > esync_timestamps[i][0])]
        signal_on_timestamps.append(signal_on_temp)

    
    timestamps.append(spike_times[np.where(spike_times < ses_e_end[0] + 100000)])
    spike_clusters2.append(spike_clusters[np.where(spike_times < ses_e_end[0] + 100000)])
    for i in range(1, Nses):
        spike_temp = spike_times[np.where(spike_times < ses_e_end[i] + 100000)] 
        cluster_temp = spike_clusters[np.where(spike_times < ses_e_end[i] + 100000)]
        cluster_temp = cluster_temp[np.where(spike_temp > ses_e_end[i-1] + 100000)]
        spike_temp = spike_temp[np.where(spike_temp > ses_e_end[i-1] + 100000)]
        timestamps.append(spike_temp)
        spike_clusters2.append(cluster_temp)
    print('sessions ended at timestamps,', ses_e_end)
    
        
    
    return spike_clusters2, timestamps, clusters_quality, vsync, esync_timestamps, dlc_files, signal_on_timestamps,waveforms

def load_no_implant_files(fdir, fn, Nses, experiment_tag, dlc_tail, mode):  
    # if Nses > 1, mind the rule of name.
    esync_timestamps_load = np.load(fdir.parent/'Esync_timestamps.npy')  
    signal_on_timestamps_load = np.load(fdir.parent/'Signal_on_timestamps.npy')                
    timestamps = []
    dlc_files = []
    vsync = []
    frame_state = []
    for n in range(1,Nses+1):
        dlc_temp = pd.read_hdf(fdir/(f'dlc_concatenate{n}.h5'))
        dlc_files.append(dlc_temp[dlc_tail[:-12]])
        # '''
        # framestate需要好好去问一下
        # '''
        if mode == 'FrameState':
            frame_state = pd.read_csv(fdir/(f'vsync{n}.csv'))    #Try using this rule to name files.
            vsync_temp = np.array(frame_state['blockID'], dtype='uint')
            vsync.append(vsync_temp)
        elif mode == 'Bonsai':
            vsync_csv = pd.read_csv(fdir/(f'vsync{n}.csv'), names=[0,1,2])
            vsync_temp = np.array(vsync_csv.loc[:,1], dtype='uint')
            vsync.append(np.where((vsync_temp[1:] - vsync_temp[:-1]) ==1)[0])
   
    # arbituarily more than 10s interval would be made when concatenate ephys files.
    
    ses_e_end = esync_timestamps_load[np.where((esync_timestamps_load[1:] - esync_timestamps_load[:-1]) > 100000)[0]]
    ses_e_end = np.append(ses_e_end, esync_timestamps_load[-1])# last one sync needed here.
    esync_timestamps = [esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[0] + 100000)]]
    
    for i in range(1, Nses):
        esync_temp = esync_timestamps_load[np.where(esync_timestamps_load < ses_e_end[i] + 100000)]
        esync_temp = esync_temp[np.where(esync_temp > ses_e_end[i-1])]
        esync_timestamps.append(esync_temp)
        
    
    signal_on_timestamps = []
    for i in range(Nses):
        signal_on_temp = signal_on_timestamps_load[np.where(signal_on_timestamps_load < esync_timestamps[i][-1])]
        signal_on_temp = signal_on_temp[np.where(signal_on_temp > esync_timestamps[i][0])]
        signal_on_timestamps.append(signal_on_temp)       
    return vsync, esync_timestamps, dlc_files, signal_on_timestamps

def sync_check(esync_timestamps, vsync, Nses, fontsize,mode):    
    for i in range(Nses):
        if mode=='Bonsai':
            vsync_temp=np.size(vsync[i])
            print(f"vsync={vsync_temp}")
        if mode=='FrameState':
            vsync_temp=np.max(vsync[i])
            print(f"vsync={vsync_temp}")
        print(f"esync_timestamps={np.size(esync_timestamps[i])}")
        if vsync_temp != np.size(esync_timestamps[i]):
            raise Exception('N of E&V Syncs do not Equal!!! Problems with Sync in ses ', str(i), '!!!')
        else:
            print('ses ', str(i),' N of E&V Syncs equal. You may continue.')
            if  mode=='Bonsai':
                fig = plt.figure(figsize=(10,5))
                ax1 = fig.add_subplot(1,2,1)
                ax2 = fig.add_subplot(1,2,2)
                ax1.set_title('N samples between Esyncs', fontsize=fontsize*1.3)
                ax2.set_title('N frames between Vsyncs', fontsize=fontsize*1.3)
                # legend?
                esync_inter = esync_timestamps[i][1:] - esync_timestamps[i][:-1]
                vsync_inter = vsync[i][1:] - vsync[i][:-1]
                ax1.hist(esync_inter, bins = len(set(esync_inter)), alpha=0.2)
                ax2.hist(vsync_inter, bins = len(set(vsync_inter)), alpha=0.2)
                plt.show()
        

'''
新的，仔细看看
下面两个函数是将timestamps，spiketime，spikecluster和signaltime转成线性插值器。
'''
def sync_cut_stamps2time(spike_clusters, timestamps, ses, esync_timestamps, sync_rate, experiment_tag):
    # head&tail cut here, then transform into frame_id for spd_mask.
    spike_clusters = np.delete(spike_clusters, np.where(timestamps > esync_timestamps[-1])[0])
    spike_clusters = np.delete(spike_clusters, np.where(timestamps < esync_timestamps[0])[0])
    timestamps = np.delete(timestamps, np.where(timestamps > esync_timestamps[-1])[0])
    timestamps = np.delete(timestamps, np.where(timestamps < esync_timestamps[0])[0])
    # assign time-values from esync_timestamps
    interp_y = np.linspace(0, (np.size(esync_timestamps)-1)/sync_rate, num=np.size(esync_timestamps))
    stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps, interp_y, k=1, s=0)
    spiketime = stamps2time_interp(timestamps)
    return (spike_clusters,timestamps,spiketime)       

def signal_stamps2time(esync_timestamps, signal_on_timestamps, Nses, sync_rate):
    signal_on_time = []
    for i in range(Nses):
        if np.size(signal_on_timestamps[i]) > 0:
            interp_y = np.linspace(0, (np.size(esync_timestamps[i])-1)/sync_rate, num=np.size(esync_timestamps[i]))
            stamps2time_interp = interpolate.UnivariateSpline(esync_timestamps[i], interp_y, k=1, s=0)
            signal_on_time.append(stamps2time_interp(signal_on_timestamps[i]))
        else:
            signal_on_time.append(np.array([]))
    return signal_on_time

# def apply_spd_mask_20msbin(spike_clusters, timestamps, spiketime, ses,temporal_bin_length=0.02)
#     # applying spd_mask means sort spikes into running and staying.
#     # if 'spatial' in experiment_tag:
#         spiketime_bin_id = (spiketime/temporal_bin_length).astype('uint')
#         spike_spd_id = ses.spd_mask[spiketime_bin_id]
#         high_spd_id = np.where(spike_spd_id==1)[0]
#         low_spd_id= np.where(spike_spd_id==0)[0]
        
#         spike_clusters_stay = spike_clusters[low_spd_id]
#         timestamps_stay = timestamps[low_spd_id]
#         spiketime_stay = spiketime[low_spd_id]
#         spike_clusters = spike_clusters[high_spd_id]
#         timestamps = timestamps[high_spd_id]
#         spiketime = spiketime[high_spd_id]
#         return (spike_clusters,timestamps,spiketime, spike_clusters_stay,timestamps_stay,spiketime_stay)
# def spatial_information_skaggs(timestamps, ratemap, dwell_smo):
#     global_mean_rate = round(np.size(timestamps)/np.sum(dwell_smo), 4)
#     spatial_info = round(np.nansum((dwell_smo/np.sum(dwell_smo)) * (ratemap/global_mean_rate) * np.log2((ratemap/global_mean_rate))), 4)
#     return spatial_info, global_mean_rate

# def time2hd(t, time2xy_interp):
#     right = np.vstack((time2xy_interp[4](t), time2xy_interp[5](t)))
#     left = np.vstack((time2xy_interp[2](t), time2xy_interp[3](t)))
#     hd_vector = right - left
#     hd_radius = np.angle(hd_vector[:,0] + 1j*hd_vector[:,1])
#     hd_degree = (hd_radius+np.pi)/(np.pi*2)*360
#     return hd_degree
    

# ----------------------------------------------------------------------------
#                  Functions on 1D Env
# ----------------------------------------------------------------------------        
# def find_center_circular_track(x,y, fontsize):#不用看
#     # try another way, circular fitting
#     # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
#     def calc_R(xc, yc):
#         """ calculate the distance of each 2D points from the center (xc, yc) """
#         return np.sqrt((x-xc)**2 + (y-yc)**2)

#     def f_2(c):
#         """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
#         Ri = calc_R(*c)
#         return Ri - Ri.mean()

#     center_estimate = np.mean(x), np.mean(y)
#     center_2,ier = optimize.leastsq(f_2, center_estimate)# could get ier or mesg, for more info of output.
#     xc_2, yc_2 = center_2
#     Ri_2       = calc_R(*center_2)
#     R_2        = Ri_2.mean()
#     residu_2   = sum((Ri_2 - R_2)**2)
#     #plot for check?
#     fig = plt.figure(figsize=(5,5))
#     ax1 = fig.add_subplot(111)
#     ax1.set_title('scatter of spatial occupancy in pixels.', fontsize=fontsize)
#     ax1.scatter(np.append(x, center_2[0]), np.append(y,center_2[1]), s=3, alpha=0.1)
#     return center_2, R_2, residu_2       


# def boxcar_smooth_1d_circular(arr, kernel_width=20):#不用看
#     arr_smo = np.convolve(np.array([1/kernel_width]*kernel_width),
#                           np.concatenate((arr,arr)))[kernel_width : (arr.shape[0]+kernel_width)]
#     arr_smo = np.concatenate((arr_smo[-(kernel_width):], arr_smo[:-(kernel_width)]))
#     return arr_smo


# def ratemap_1d_circular(spiketime, time2xy_interp, dwell_smo, nspatial_bins):#不用看
    
#     spk_pol = np.angle(time2xy_interp[0](spiketime) + 1j*time2xy_interp[1](spiketime))
#     spk_bin = ((spk_pol+np.pi)/(2*np.pi)*nspatial_bins).astype('uint')
#     Nspike_in_bins = npg.aggregate(spk_bin, 1, size=nspatial_bins)
#     Nspike_in_bins_smo = boxcar_smooth_1d_circular(Nspike_in_bins)
#     ratemap = Nspike_in_bins_smo/dwell_smo
#     return ratemap


# def positional_information_olypher_1dcircular(spiketime, time2xy_interp, total_time, temporal_bin_length, nspatial_bins):#看自己写的

#     t = np.linspace(0, total_time, num=(total_time/temporal_bin_length).astype('uint'), endpoint=False) + 0.5*temporal_bin_length#assign spatial values at midpoint of every time bin.
#     pol = np.angle(time2xy_interp[0](t) + 1j*time2xy_interp[1](t))
#     pol_temporal_bin = ((pol+np.pi)/(2*np.pi)*nspatial_bins).astype('uint')
    
#     spk_time_temp = (spiketime/temporal_bin_length).astype('uint')
#     # emmm this would get nearest up or down???
    
#     spk_count_temporal_bin = npg.aggregate(spk_time_temp, 1, size=(total_time/temporal_bin_length).astype('uint'))
#     p_k = npg.aggregate(spk_count_temporal_bin,1)/np.sum(npg.aggregate(spk_count_temporal_bin,1))
    
#     pos_info = []
#     for i in range(nspatial_bins):
#         spk_count_temporal_bin_xi = spk_count_temporal_bin[np.where(pol_temporal_bin==i)]
#         p_kxi = npg.aggregate(spk_count_temporal_bin_xi, 1)/np.sum(npg.aggregate(spk_count_temporal_bin_xi, 1))
#         pos_info.append(np.sum(p_kxi * np.log2(p_kxi/p_k[:np.size(p_kxi)])))# set range for p_k is that, e.g. some time bin might have 8 spks or more but not in certain spatial bin, then arrays are not the same length. 
#     return np.array(pos_info)
        
    
def shuffle_test_1d_circular(u, session, Nses, temporal_bin_length=0.02, nspatial_bins_spa=360, nspatial_bins_pos=48, p_threshold=0.01):
    # Ref, Monaco2014, head scanning, JJKnierim's paper.
    # Units must pass shuffle test and either their spa_info >1 or max_pos_info >0.4
    
    # not working well so far.
    '''
    maybe useful. 
    '''
    
    units = []
    for i in u:
        if u.type == 'excitatory':
            units.append(i)
    spatial_info_pool = []
    positional_info_pool = []
    if Nses == 1:
        for i in units:
            spiketime = np.hstack((i.spiketime, i.spiketime_stay))
            spiketime = session.total_time - spiketime# invert
            spiketime.sort()
            for k in range(1000):
                spiketime += (session.total_time-8) * random.random() + 4 #offset should be at least 4s away from start/end of the session.
                spiketime[np.where(spiketime>session.total_time)] = spiketime[np.where(spiketime>session.total_time)] - session.total_time# wrap back.
                # spd_masking
                spiketime_bin_id = (spiketime/temporal_bin_length).astype('uint')
                spiketime_run = spiketime[session.spd_mask[spiketime_bin_id]]
                # spatial info
                ratemap = ratemap_1d_circular(spiketime_run, session.time2xy_interp, session.dwell_smo, nspatial_bins_spa)
                spa_info = spatial_information_skaggs(spiketime_run, ratemap, session.dwell_smo)
                spatial_info_pool.append(spa_info)
                # positional info
                pos_info = positional_information_olypher_1dcircular(spiketime_run, session.time2xy_interp, session.total_time, temporal_bin_length, nspatial_bins_pos)
                positional_info_pool.append(np.nanmax(pos_info))
                
        spatial_info_pool = np.sort(np.array(spatial_info_pool))
        positional_info_pool = np.sort(np.array(positional_info_pool))
        shuffle_bar_spa = spatial_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        shuffle_bar_pos = positional_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        print('Shuffle results: spatial_info {0}, postional_info {1}'.format(round(shuffle_bar_spa,4), round(shuffle_bar_pos,4)))
        for i in units:
            if i.spatial_info>shuffle_bar_spa and i.positional_info>shuffle_bar_pos:
                if i.spatial_info>1 or i.postional_info>0.4:
                    i.is_place_cell = True
    if Nses > 1:
        for i in units:
            for j in session:
                spiketime = np.hstack((i.spiketime[j.id], i.spiketime_stay[j.id]))
                spiketime = j.total_time - spiketime# invert
                spiketime.sort()
                for k in range(1000):
                    spiketime += (j.total_time-8) * random.random() + 4 #offset should be at least 4s away from start/end of the session.
                    spiketime[np.where(spiketime>j.total_time)] = spiketime[np.where(spiketime>j.total_time)] - j.total_time# wrap back.
                    # spd_masking
                    spiketime_bin_id = (spiketime/temporal_bin_length).astype('uint')
                    spiketime_run = spiketime[j.spd_mask[spiketime_bin_id]]
                    # spatial info
                    ratemap = ratemap_1d_circular(spiketime_run, j.time2xy_interp, j.dwell_smo, nspatial_bins_spa)
                    spa_info = spatial_information_skaggs(spiketime_run, ratemap, j.dwell_smo)
                    spatial_info_pool.append(spa_info)
                    # positional info
                    pos_info = positional_information_olypher_1dcircular(spiketime_run, j.time2xy_interp, j.total_time, temporal_bin_length, nspatial_bins_pos)
                    positional_info_pool.append(np.nanmax(pos_info))
                    
                    
        spatial_info_pool = np.sort(np.array(spatial_info_pool))
        positional_info_pool = np.sort(np.array(positional_info_pool))
        shuffle_bar_spa = spatial_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        shuffle_bar_pos = positional_info_pool[int(np.size(spatial_info_pool)*p_threshold) * (-1)]
        print('Shuffle results: spatial_info {0}, postional_info {1}'.format(round(shuffle_bar_spa,4), round(shuffle_bar_pos,4)))
        for i in units:
            for j in session:
                if i.spatial_info[j.id]>shuffle_bar_spa and i.positional_info[j.id]>shuffle_bar_pos:
                    if i.spatial_info[j.id]>1 or i.postional_info[j.id]>0.4:
                        i.is_place_cell[j.id] = True


# ----------------------------------------------------------------------------
#                  Functions on 2D Env
# ----------------------------------------------------------------------------
# def boxcar_smooth_2d():
#     pass

# def gaussian_smooth_2d():
#     pass   

# def shuffle_test_2d():
#     # this would be simple, just go with random temporal offset and play around 1000 times. No worries like in 1d.
#     pass

# def Kalman_filter_2d():#interpreted from Bohua's code.
#     pass

# def time2xy(ses,t):
#     return np.array(ses.time2xy_interp[0](t), (ses.time2xy_interp[1](t)))
# ----------------------------------------------------------------------------
#                  Classes session
# ----------------------------------------------------------------------------

class Session(object):    
    def __init__(self, dlch5, dlc_col_ind_dict, vsync, sync_rate, experiment_tag,
                 ses_id=0, fontsize=15):
        # docstring?           
        self.id = ses_id
        self.vsync = vsync
        self.experiment_tag = experiment_tag
        self.sync_rate = sync_rate
        self.fontsize = fontsize
        # self.pixpcm = 0
        self.cut = False
        
        '''
        这部分为了筛选pos方便，考虑直接吧筛选位置写进来，参考NCP_lty
        另外，去问一下Framestate那几个啥意思哦
        '''
        
        if 'Bonsai' in self.experiment_tag:
            self.left_pos = np.vstack((np.array(dlch5[dlch5.columns[dlc_col_ind_dict['left_pos']]]), np.array(dlch5[dlch5.columns[dlc_col_ind_dict['left_pos']+1]]))).T
            self.right_pos = np.vstack((np.array(dlch5[dlch5.columns[dlc_col_ind_dict['right_pos']]]), np.array(dlch5[dlch5.columns[dlc_col_ind_dict['right_pos']+1]]))).T 
        elif 'FrameState' in self.experiment_tag:
            dlcmodelstr=dlch5.columns[1][0]
            for key in dlc_col_ind_dict:# for customized need from DLC.
                pos_for_key = np.vstack((np.array(dlch5[(dlcmodelstr,key,'x')]), np.array(dlch5[(dlcmodelstr,key,'y')]))).T    
                setattr(self, key, pos_for_key)
            for key in frame_state.columns:
                setattr(self, key, np.array(frame_state[key]).T)
            self.raw_frame_length = (getattr(self, 'Frame')).shape[0]       #mark down the total frame length of raw video, for sync cut
                

    def sync_cut_generate_frame_time(self):
        if 'Bonsai' in self.experiment_tag:
            if self.cut == False:
                self.left_pos = self.left_pos[self.vsync[0]:self.vsync[-1]+1, :]
                self.right_pos = self.right_pos[self.vsync[0]:self.vsync[-1]+1, :]
                #assign time values for frames. So far for a single ses, single video.
                frame2time_interp = interpolate.UnivariateSpline(self.vsync-self.vsync[0], np.linspace(0, (np.size(self.vsync)-1)/self.sync_rate, num=np.size(self.vsync)),
                                                                 k=1, s=0)
                self.frame_time = frame2time_interp(np.arange(self.left_pos.shape[0])).astype('float64')
                self.total_time = self.frame_time[-1]
                self.cut = True
            else:
                print('Has already being sync_cut, noway to do a second time.')
        elif 'FrameState' in experiment_tag:
            for key in vars(self):
                FrameData = getattr(self, key)
                if hasattr(FrameData, 'shape') and FrameData.shape[0] == self.raw_frame_length:           #if the data length quals to raw video's , it needs sync cut head tail 
                    FrameData = FrameData[self.vsync[0]:self.vsync[-1]+1]
                    setattr(self, key, FrameData)
            self.frame_length = len(self.Frame)
            frame2time_interp = interpolate.UnivariateSpline(self.vsync-self.vsync[0], np.linspace(0, (np.size(self.vsync)-1)/self.sync_rate, num=np.size(self.vsync)),k=1, s=0)
            self.frame_time = frame2time_interp(np.arange(self.framelength)).astype('float64')
            self.total_time = self.frame_time[-1]
        else:
            raise Exception('Please choose your V_recording mode.')
            

    def remove_nan_merge_pos_get_hd(self):        
        nan_id = np.isnan(self.left_pos) + np.isnan(self.right_pos)
        nan_id = nan_id[:,0] + nan_id[:,1]
        nan_id = np.where(nan_id == 2, 1, 0).astype('bool')
        self.frame_time = self.frame_time[~nan_id]
        self.left_pos = self.left_pos[~nan_id]
        self.right_pos = self.right_pos[~nan_id]      
        # if 'spatial' in self.experiment_tag:
        #     hd_vector = self.right_pos - self.left_pos
        #     hd_radius = np.angle(hd_vector[:,0] + 1j*hd_vector[:,1])
        #     self.hd_degree = (hd_radius+np.pi)/(np.pi*2)*360
            
        
        self.pos_pix = (self.left_pos + self.right_pos)/2
        if 'circular' in self.experiment_tag:
            self.pos = ((self.left_pos + self.right_pos)/2 - self.center[0])/self.pixpcm
        else:
            print('you need to code your way do define pixels per cm, to go furthur.')
    
    def generate_time2xy_interpolate(self, mode='linear'):
        if mode == 'cspline':
            # using scipy.interpolate.UnivariateSpline seperately with x&y might cause some infidelity. Mind this.
            # k = 3, how to set a proper smooth factor s???
            # what about it after Kalman filter?
            # how to check this???
            time2x_interp = interpolate.UnivariateSpline(self.frame_time, self.pos[:,0])
            time2y_interp = interpolate.UnivariateSpline(self.frame_time, self.pos[:,1])
            time2x_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,0])
            time2y_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,1])
            time2x_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,0])
            time2y_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,1])
        elif mode == 'linear':
            time2x_interp = interpolate.UnivariateSpline(self.frame_time, self.pos[:,0], k=1)
            time2y_interp = interpolate.UnivariateSpline(self.frame_time, self.pos[:,1], k=1)
            time2x_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,0], k=1)
            time2y_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,1], k=1)
            time2x_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,0], k=1)
            time2y_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,1], k=1)
        self.time2xy_interp = (time2x_interp, time2y_interp, time2x_left, time2y_left, time2x_right, time2y_right)


    def generate_spd_mask_20ms_bin(self, threshold=2, temporal_bin_length=0.02):
        t = np.linspace(0, self.total_time, num=(self.total_time/temporal_bin_length +1).astype('uint'))
        x = self.time2xy_interp[0](t)
        y = self.time2xy_interp[1](t)
        dist = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        self.inst_spd = dist/temporal_bin_length
        self.spd_mask = np.where(self.inst_spd > 2, 1, 0)
        self.spd_mask = np.append(self.spd_mask, 0).astype('bool')# for convinience.


class Social_session(object):
    def __init__(self, dlc_files, cage_line,vsync, sync_rate, experiment_tag,tracksize,mode,
                 ses_id=0,filter_time=3, fontsize=15):
        # docstring?           
        self.id = ses_id
        self.cage_line=cage_line
        self.vsync = vsync
        self.experiment_tag = experiment_tag
        self.sync_rate = sync_rate
        self.fontsize = fontsize
        # self.pixpcm = 0
        self.cut = False
        self.mode=mode
        self.tracksize=tracksize
        self.filter_time=filter_time
        if self.mode == 'Bonsai':
            # self.left_pos = np.vstack((np.array(dlch5[dlch5.columns[dlc_col_ind_dict['left_pos']]]), np.array(dlch5[dlch5.columns[dlc_col_ind_dict['left_pos']+1]]))).T
            # self.right_pos = np.vstack((np.array(dlch5[dlch5.columns[dlc_col_ind_dict['right_pos']]]), np.array(dlch5[dlch5.columns[dlc_col_ind_dict['right_pos']+1]]))).T 
            self.dlc_files=dlc_files
            
        elif mode =='FrameState':
            self.dlc_files=dlc_files
            # dlcmodelstr=dlch5.columns[1][0]
            # for key in dlc_col_ind_dict:# for customized need from DLC.
            #     pos_for_key = np.vstack((np.array(dlch5[(dlcmodelstr,key,'x')]), np.array(dlch5[(dlcmodelstr,key,'y')]))).T    
            #     setattr(self, key, pos_for_key)
            # for key in frame_state.columns:
            #     setattr(self, key, np.array(frame_state[key]).T)
            # self.raw_frame_length = (getattr(self, 'Frame')).shape[0]       #mark down the total frame length of raw video, for sync cut
    
    def sync_cut_generate_frame_time(self):
        if self.mode == 'Bonsai':
            if self.cut == False:
                self.dlc_files= self.dlc_files.iloc[self.vsync[0]:self.vsync[-1]+1]
                #assign time values for frames. So far for a single ses, single video.
                frame2time_interp = interpolate.UnivariateSpline(self.vsync-self.vsync[0], np.linspace(0, (np.size(self.vsync)-1)/self.sync_rate, num=np.size(self.vsync)),
                                                                 k=1, s=0)
                self.frame_time = frame2time_interp(np.arange(self.dlc_files.shape[0])).astype('float64')
                self.total_time = self.frame_time[-1]
                self.cut = True
            else:
                print('Has already being sync_cut, noway to do a second time.')
        elif self.mode == 'FrameState':
        #     for key in vars(self):
        #         FrameData = getattr(self, key)
        #         if hasattr(FrameData, 'shape') and FrameData.shape[0] == self.raw_frame_length:           #if the data length quals to raw video's , it needs sync cut head tail 
        #             FrameData = FrameData[self.vsync[0]:self.vsync[-1]+1]
        #             setattr(self, key, FrameData)
        #     self.frame_length = len(self.Frame)
        #     frame2time_interp = interpolate.UnivariateSpline(self.vsync-self.vsync[0], np.linspace(0, (np.size(self.vsync)-1)/self.sync_rate, num=np.size(self.vsync)),k=1, s=0)
        #     self.frame_time = frame2time_interp(np.arange(self.framelength)).astype('float64')
        #     self.total_time = self.frame_time[-1]
        # else:
        #     raise Exception('Please choose your V_recording mode.')
            if self.cut == False:
                self.dlc_files= self.dlc_files.iloc[self.vsync[0]:self.vsync[-1]+1]
                #assign time values for frames. So far for a single ses, single video.
                frame2time_interp = interpolate.UnivariateSpline(self.vsync-self.vsync[0], np.linspace(0, (np.size(self.vsync)-1)/self.sync_rate, num=np.size(self.vsync)),
                                                                 k=1, s=0)
                self.frame_time = frame2time_interp(np.arange(self.dlc_files.shape[0])).astype('float64')
                self.total_time = self.frame_time[-1]
                self.cut = True
            else:
                print('Has already being sync_cut, noway to do a second time.')
    
    def filter_pos(self,landmark_num=8):
        self.dlc_filter=self.dlc_files.loc[:, (self.dlc_files.columns.get_level_values(1)!='likelihood')]
        landmark_pos_df=self.dlc_files.iloc[:,0:landmark_num*3]
        self.landmark_pos=pd.DataFrame()
        for corner in ["corner1", "corner2", "corner3", "corner4"]:
            # 筛选出likelihood大于0.9的行
            filtered = landmark_pos_df[corner].query("likelihood > 0.99")
            # 计算x和y坐标的平均值，并添加到结果DataFrame中
            self.landmark_pos[corner] = filtered[["x", "y"]].mean(axis=0)
        for cage in ["cage1", "cage2", "cage3", "cage4"]:
            # 筛选出likelihood大于0.9的行
            filtered = landmark_pos_df[cage].query("likelihood > 0.9")
            # 计算x和y坐标的平均值，并添加到结果DataFrame中
            self.landmark_pos[cage] = filtered[["x", "y"]].mean(axis=0)
        self.cage_pos=self.dlc_filter.iloc[:,landmark_num:(2*landmark_num)].apply(np.mean,axis=0)
        self.dlc_filter=self.dlc_filter.iloc[:,(2*landmark_num):]
        self.pos=(self.dlc_filter['leftled']+self.dlc_filter['rightled'])/2
        # self.vector=self.dlc_filter['leftled']-self.dlc_filter['rightled']
        self.mask=(self.dlc_filter-self.pos)
        self.mask=self.dlc_filter.iloc[:,8:]
        # self.pos=pd.concat((self.pos,self.dlc_filter['leftled'],self.dlc_filter['rightled']),axis=1)
        
        def square_sum(df):
            # 获取所有列名
            cols = df.columns
            # 生成索引序列，每隔2步
            idx = range(0, len(cols), 2)
            # 创建一个空列表，用于存储每两列的求平方和结果
            res = []
            # 遍历索引序列
            for i in idx:
            # 取出两个相邻的列名
                col1 = cols[i]
                col2 = cols[i+1]
                # 选取对应的两列数据
                data = df[[col1, col2]]
                # 对数据进行平方运算
                data = np.square(data)
                # 对数据进行求和运算，得到一个Series对象
                data = np.sum(data, axis=1)
                # 将Series对象添加到列表中
                res.append(data)
              # 将列表中的所有Series对象沿着列方向拼接起来，得到一个新的DataFrame对象
            res = pd.concat(res, axis=1)
          # 返回结果
            return res
        self.mask=square_sum(self.mask)
        def quantile_bool (series):
            q90 = series.quantile (0.9) # 计算 90 分位数
            q10 = series.quantile (0.1) # 计算 10 分位数
            return (series >= 0.8*q10) & (series <= 1.2*q90) # 返回布尔值
        self.mask=self.mask.apply(quantile_bool,axis=0)
        self.mask=self.mask.apply(sum,axis=1)
        self.mask[self.mask.index[0]]=1
        self.pos[self.mask==0]=np.nan
        self.pos=self.pos.interpolate('linear',axis=0)
        #到这一步做完了position数据的筛选，后面需要给他转换成只在箱子里的数据。
        max_df=self.landmark_pos.max(axis=1).to_frame('max')
        min_df=self.landmark_pos.min(axis=1).to_frame('min')
        self.landmark_pos= max_df.join(min_df)
        self.pixpcm=(self.landmark_pos['max']-self.landmark_pos['min'])/self.tracksize
        self.pos2=(self.pos-self.landmark_pos['min'])/self.pixpcm
        self.pos2['ledleft_x']=(self.dlc_filter['leftled']['x']-self.landmark_pos['min']['x'])/self.pixpcm['x']
        self.pos2['ledleft_y']=(self.dlc_filter['leftled']['y']-self.landmark_pos['min']['y'])/self.pixpcm['y']
        self.pos2['ledright_x']=(self.dlc_filter['rightled']['x']-self.landmark_pos['min']['x'])/self.pixpcm['x']
        self.pos2['ledright_y']=(self.dlc_filter['rightled']['y']-self.landmark_pos['min']['y'])/self.pixpcm['y']
        self.pos2[self.pos2>50]=50
        self.pos2[self.pos2<0]=0
        self.cage_pos2=(self.cage_pos-self.landmark_pos['min'])/self.pixpcm
        self.cage_pos2['cage1']['x']=4
        self.cage_pos2['cage1']['y']=4
        self.cage_pos2['cage2']['x']=46
        self.cage_pos2['cage2']['y']=4
        self.cage_pos2['cage3']['x']=4
        self.cage_pos2['cage3']['y']=46
        self.cage_pos2['cage4']['x']=46
        self.cage_pos2['cage4']['y']=46
    def get_path_filter(self):
        dx = np.diff(self.pos2['x']) # 计算x坐标的差值
        dy = np.diff(self.pos2['x']) 
        self.dist = np.linalg.norm(np.vstack([dx, dy]), axis=0) # 计算欧几里得距离
        self.pos2['dist'] = np.insert(self.dist, 0, 0) # 在距离数组的开头添加一个NaN值
        self.pos2[self.pos2['dist']>np.quantile(self.dist,0.9)]=np.nan
        self.pos2=self.pos2.interpolate('linear',axis=0)
        dx = np.diff(self.pos2['x']) # 计算x坐标的差值
        dy = np.diff(self.pos2['x']) 
        self.dist = np.linalg.norm(np.vstack([dx, dy]), axis=0) # 计算欧几里得距离
        self.pos2['dist'] = np.insert(self.dist, 0, 0) # 在距离数组的开头添加一个NaN值
        self.pos2['path'] = self.pos2['dist'].cumsum()
        
    def filter_pos_noimplant(self,landmark_num=8):
        self.dlc_filter=self.dlc_files.loc[:, (self.dlc_files.columns.get_level_values(1)!='likelihood')]
        landmark_pos_df=self.dlc_files.iloc[:,0:landmark_num*3]
        self.landmark_pos=pd.DataFrame()
        for corner in ["corner1", "corner2", "corner3", "corner4"]:
            # 筛选出likelihood大于0.9的行
            filtered = landmark_pos_df[corner].query("likelihood > 0.99")
            # 计算x和y坐标的平均值，并添加到结果DataFrame中
            self.landmark_pos[corner] = filtered[["x", "y"]].mean(axis=0)
        self.cage_pos=self.dlc_filter.iloc[:,landmark_num:(2*landmark_num)].apply(np.mean,axis=0)
        self.dlc_filter=self.dlc_filter.iloc[:,(2*landmark_num):]
        self.pos=(self.dlc_filter['leftear']+self.dlc_filter['rightear'])/2
        self.mask=(self.dlc_filter-self.pos)
        self.mask=self.dlc_filter.iloc[:,8:]
        
        def square_sum(df):
            # 获取所有列名
            cols = df.columns
            # 生成索引序列，每隔2步
            idx = range(0, len(cols), 2)
            # 创建一个空列表，用于存储每两列的求平方和结果
            res = []
            # 遍历索引序列
            for i in idx:
            # 取出两个相邻的列名
                col1 = cols[i]
                col2 = cols[i+1]
                # 选取对应的两列数据
                data = df[[col1, col2]]
                # 对数据进行平方运算
                data = np.square(data)
                # 对数据进行求和运算，得到一个Series对象
                data = np.sum(data, axis=1)
                # 将Series对象添加到列表中
                res.append(data)
              # 将列表中的所有Series对象沿着列方向拼接起来，得到一个新的DataFrame对象
            res = pd.concat(res, axis=1)
          # 返回结果
            return res
        self.mask=square_sum(self.mask)
        def quantile_bool (series):
            q90 = series.quantile (0.9) # 计算 90 分位数
            q10 = series.quantile (0.1) # 计算 10 分位数
            return (series >= 0.8*q10) & (series <= 1.2*q90) # 返回布尔值
        self.mask=self.mask.apply(quantile_bool,axis=0)
        self.mask=self.mask.apply(sum,axis=1)
        self.mask[1]=1
        self.pos[self.mask==0]=np.nan
        self.pos=self.pos.interpolate()
        #到这一步做完了position数据的筛选，后面需要给他转换成只在箱子里的数据。
        max_df=self.landmark_pos.max(axis=1).to_frame('max')
        min_df=self.landmark_pos.min(axis=1).to_frame('min')
        self.landmark_pos= max_df.join(min_df)
        self.pixpcm=(self.landmark_pos['max']-self.landmark_pos['min'])/self.tracksize
        self.pos2=(self.pos-self.landmark_pos['min'])/self.pixpcm
        self.pos2[self.pos2>50]=50
        self.pos2[self.pos2<0]=0
        self.cage_pos2_dlc=(self.cage_pos-self.landmark_pos['min'])/self.pixpcm
        self.cage_pos2 = self.cage_pos2_dlc
        self.cage_pos2['cage1']['x']=4
        self.cage_pos2['cage1']['y']=4
        self.cage_pos2['cage2']['x']=46
        self.cage_pos2['cage2']['y']=4
        self.cage_pos2['cage3']['x']=4
        self.cage_pos2['cage3']['y']=46
        self.cage_pos2['cage4']['x']=46
        self.cage_pos2['cage4']['y']=46
    def plot_track(self):
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot()
        ax1.plot(self.pos2['x'],self.pos2['y'],zorder=1)
        ax1.scatter(self.cage_pos2.xs ('x', level='coords').to_numpy (),\
                    self.cage_pos2.xs ('y', level='coords').to_numpy (),\
                        c='blue', s=4000,zorder=1)
        
        # plt.title(f'track for {self.experiment_tag}')
        plt.xlabel("track")
        # plt.savefig(fpath_fig/f'{unit}//scatter plot for ses{ses_n+1} unit{unit}.png')
        # plt.show()
    def generate_time2xy_interpolate(self, mode='linear'):
        if mode == 'cspline':
            # using scipy.interpolate.UnivariateSpline seperately with x&y might cause some infidelity. Mind this.
            # k = 3, how to set a proper smooth factor s???
            # what about it after Kalman filter?
            # how to check this???
            time2x_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['x'])
            time2y_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['y'])
            # time2x_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,0])
            # time2y_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,1])
            # time2x_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,0])
            # time2y_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,1])
        elif mode == 'linear':
            time2x_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['x'], k=1)
            time2y_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['y'], k=1)
            # time2ledleft_x_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['ledleft_x'], k=1)
            # time2ledleft_y_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['ledleft_y'], k=1)
            # time2ledright_x_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['ledright_x'], k=1)
            # time2ledright_y_interp = interpolate.UnivariateSpline(self.frame_time, self.pos2['ledright_y'], k=1)
            # time2x_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,0], k=1)
            # time2y_left = interpolate.UnivariateSpline(self.frame_time, self.left_pos[:,1], k=1)
            # time2x_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,0], k=1)
            # time2y_right = interpolate.UnivariateSpline(self.frame_time, self.right_pos[:,1], k=1)
        # self.time2xy_interp = (time2x_interp, time2y_interp, time2x_left, time2y_left, time2x_right, time2y_right)
        # self.time2xy_interp = (time2x_interp, time2y_interp,time2ledleft_x_interp,time2ledleft_y_interp,time2ledright_x_interp,time2ledright_y_interp)
        self.time2xy_interp = (time2x_interp, time2y_interp)
    def get_social_mark(self,dist=12):
        def distance(x1, y1, x2, y2):
          # 计算两个点之间的欧几里得距离
          return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        def label(row):
          # 根据距离判断标签
          x = row['x']
          y = row['y']
          if distance(x, y, self.cage_pos2['cage1','x'],self.cage_pos2['cage1','y']) < dist:
            return self.cage_line[0]
          elif distance(x, y, self.cage_pos2['cage2','x'],self.cage_pos2['cage2','y']) < dist:
            return self.cage_line[1]
          elif distance(x, y, self.cage_pos2['cage3','x'],self.cage_pos2['cage3','y']) < dist:
            return self.cage_line[2]
          elif distance(x, y, self.cage_pos2['cage4','x'],self.cage_pos2['cage4','y']) < dist:
            return self.cage_line[3]
          else:
            return 0

        def mark(df):
          # 对DataFrame的每一行应用label函数，得到一个新的Series对象
          s = df.apply(label, axis=1)
          # 将新的Series对象作为新的一列添加到DataFrame中，命名为label
          df['label'] = s
          # 返回修改后的DataFrame
          return df
        mark(self.pos2)
        def filter_label(df,filter_time=3):
          # 对label列进行移位操作，得到前一个元素
          prev = df['label'].shift()
          # 比较label列和前一个元素，得到是否不相等的布尔值
          diff = df['label'].ne(prev)
          # 对布尔值进行累加操作，得到分组编号
          group = diff.cumsum()
          # 按照分组编号对label列进行分组，得到每个分组的大小
          size = df['label'].groupby(group).transform('size')
          # 根据分组的大小判断是否保留原始的label值，得到新的label值
          new_label = df['label'].where(size >= 60*filter_time, 0)
          # 将新的label值作为新的一列添加到DataFrame中，命名为new_label
          df['new_label'] = new_label
          # 返回修改后的DataFrame
          return df
        filter_label(self.pos2,filter_time=self.filter_time)
    def get_social_duration(self):
        self.duration_row=(self.pos2['label'].value_counts())*(self.frame_time[1]-self.frame_time[0])
        self.duration_filter=(self.pos2['new_label'].value_counts())*(self.frame_time[1]-self.frame_time[0])
        print(f'row_duration for ses{self.id}\n{self.duration_row}')
        
        print(f'filter_duration for ses{self.id}\n{self.duration_filter}')
        
    def get_dwell_map(self,binsize):
        self.pos_dwell=np.array([self.pos2['x']/binsize,self.pos2['y']/binsize]).astype('int32').T
        self.f_intervals = self.frame_time[1:] - self.frame_time[0:-1]
        self.f_int3 = np.concatenate((np.array([np.mean(self.f_intervals)]), (self.frame_time[1:] - self.frame_time[0:-1])))
        kdx,kdy = np.ogrid[-2:3,-2:3]
        gk = np.exp(-(kdx*kdx +kdy*kdy) / 2*1*1)
        gk = gk/np.sum(gk)
        # try different kernel, boxcar here.
        bck = np.ones((5,5))/(5*5)
        # dwell_smo = signal.convolve(dwell, gk, 'same', 'direct')
        self.dwell = np.flip(npg.aggregate(np.flip(self.pos_dwell.T, axis = 0), self.f_int3), axis=0)
        # self.dwell_smo = signal.convolve(self.dwell, bck, 'same', 'direct')
        mask = np.zeros_like(self.dwell)
        mask[self.dwell == 0] = True
        self.dwell_smo = gaussian_filter(self.dwell, sigma=1)
        seaborn.heatmap(self.dwell_smo,xticklabels='', yticklabels='',cmap = 'jet',vmax=10)
    def get_allo_dwell_map(self,binsize):
        self.vector=np.array([self.pos2.ledleft_x-self.pos2.ledright_x,self.pos2.ledleft_y-self.pos2.ledright_y])
        self.angle_rad=np.arctan2(self.vector[1], self.vector[0])
        self.rotation_matrix=np.array([[np.cos(self.angle_rad), -np.sin(self.angle_rad)], 
                                    [np.sin(self.angle_rad), np.cos(self.angle_rad)]])
        self.rotated_pos2=pd.DataFrame()
        self.pos_allo_dwell=pd.DataFrame()
        self.rotated_allo_dwell_map={}
        self.rotated_allo_dwell_map_smo={}
        
        for n,zone in enumerate(self.cage_line):
            pos2_temp_allo=(self.pos2.iloc[:,:2]-self.cage_pos2[f'cage{n+1}'])/binsize
            self.rotated_pos2[f'cage{n+1}'] =[np.dot(a,b) for a,b in zip(np.array(pos2_temp_allo),self.rotation_matrix.T)]
            temp_rotated_pos_x =np.array([rotated_pos[0] for rotated_pos in self.rotated_pos2[f'cage{n+1}']])
            temp_rotated_pos_y =np.array([rotated_pos[1] for rotated_pos in self.rotated_pos2[f'cage{n+1}']])
            # self.pos_allo_dwell[f'cage{zone}']={}
            self.pos_allo_dwell[f'cage{zone}','x']=temp_rotated_pos_x
            self.pos_allo_dwell[f'cage{zone}','y']=temp_rotated_pos_y
            pos_allo_dwell_temp=np.array([temp_rotated_pos_x-temp_rotated_pos_x.min(),temp_rotated_pos_y-temp_rotated_pos_y.min()]).astype('int32').T
            # self.f_intervals = self.frame_time[1:] - self.frame_time[0:-1]
            # self.f_int3 = np.concatenate((np.array([np.mean(self.f_intervals)]), (self.frame_time[1:] - self.frame_time[0:-1])))
            kdx,kdy = np.ogrid[-2:3,-2:3]
            gk = np.exp(-(kdx*kdx +kdy*kdy) / 2*1*1)
            gk = gk/np.sum(gk)
            # try different kernel, boxcar here.
            bck = np.ones((5,5))/(5*5)
            # dwell_smo = signal.convolve(dwell, gk, 'same', 'direct')
            self.rotated_allo_dwell_map[f'cage{zone}'] = np.flip(npg.aggregate(np.flip(pos_allo_dwell_temp.T, axis = 0), self.f_int3), axis=0)
            # self.dwell_smo = signal.convolve(self.dwell, bck, 'same', 'direct')
            mask = np.zeros_like(self.rotated_allo_dwell_map[f'cage{zone}'])
            mask[self.rotated_allo_dwell_map[f'cage{zone}'] == 0] = True
            self.rotated_allo_dwell_map_smo[f'cage{zone}'] = gaussian_filter(self.rotated_allo_dwell_map[f'cage{zone}'], sigma=1)
# 


def get_social_time_range(df,zone_num):
    df['inzone']= df['label'] == zone_num
    df['shifted'] = df['inzone'].shift()
    df['group'] = df.groupby((df['inzone'] != df['shifted']).cumsum()).cumcount() + 1
    df['start'] = df.groupby((df['inzone'] != df['shifted']).cumsum())['group'].transform('min')
    df['end'] = df.groupby((df['inzone'] != df['shifted']).cumsum())['group'].transform('max')
    result = df[df['inzone']][['start', 'end']].drop_duplicates()
    def add_index(row):
        # 获取行的索引值
        index = row.name
        # 将索引值加到start和end上
        row['start'] += index-1
        row['end'] += index-1
        # 返回新的行数据
        return row
    result=result.apply(add_index,axis=1)
    start_time=np.array(df['timepoint'][result['start']])
    end_time=np.array(df['timepoint'][result['end']])
    return result['start'],result['end'],start_time,end_time,end_time-start_time

  


def PSTH_invest():
    pass
def PSTH_invest_3s():
    pass
def Raster_invest():
    pass
def Raster_invest_3s():
    pass
def rate_map(fr_pos,dwell, dwell_smo, binsize=2):
    fr_pos[fr_pos<0]=0
    fr_pos=(fr_pos/binsize).astype('int32')
    
    fr_pos[:,0][fr_pos[:,0]>=(dwell.shape[0]-1)]=dwell.shape[1]-1
    fr_pos[:,1][fr_pos[:,1]>=(dwell.shape[1]-1)]=dwell.shape[0]-1
    clu_map = np.flip(npg.aggregate(np.flip(fr_pos.T), np.ones((np.size(fr_pos,axis=0)), dtype=float), size = list(np.shape(dwell_smo))), axis=0)
    # kdx,kdy = np.ogrid[-2:3,-2:3]
    # gk = np.exp(-(kdx*kdx +kdy*kdy) / 2*1*1)
    # gk = gk/np.sum(gk)
    # # try different kernel, boxcar here.
    # bck = np.ones((5,5))/(5*5)
    dwell_smo=np.where(dwell_smo==0,np.nan,dwell_smo)
    # clu_map_smo = signal.convolve(clu_map, gk, 'same', 'direct')
    # clu_map_smo = signal.convolve(clu_map, bck, 'same', 'direct')
    
    clu_map_smo = gaussian_filter(clu_map, sigma=1)
    clu_ratemap_smo = np.divide(clu_map_smo, dwell_smo) 
    clu_ratemap=np.divide(clu_map,dwell)
    min_val = np.nanmin(clu_ratemap_smo)
    max_val = np.nanmax(clu_ratemap_smo)

    # 将数组中的所有值缩放到 0 到 1 的区间
    clu_ratemap_smo_0_1 = (clu_ratemap_smo - min_val) / (max_val - min_val)
    # mask = np.zeros_like(clu_ratemap_smo)
    # mask[clu_ratemap_smo == np.nan] = 250
    mask = np.isnan(clu_ratemap_smo_0_1)

    # # 将 mask 的值设置为白色 (1.0)
    clu_ratemap_smo_0_1[mask] = np.nan
    # clu_ratemap_smo[dwell_smo == np.nan]=np.nan
    # clu_ratemap= np.nan_to_num(clu_ratemap)
    # mean_rate=np.nanmean(clu_ratemap)    
    # max_rate=np.quantile(clu_ratemap, 0.75)
    
    # spainfo = np.nansum((dwell_smo/np.sum(dwell_smo)) * (clu_ratemap/meanfiringrate) * np.log2(clu_ratemap/meanfiringrate))
    # seaborn.heatmap(clu_ratemap, xticklabels='', yticklabels='',cmap = 'jet',center=mean_rate,vmin=0,vmax=2*mean_rate)
    # plt.show()
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(clu_ratemap_smo_0_1, mask=mask,xticklabels='', yticklabels='',cmap = 'jet')
    # seaborn.heatmap(clu_ratemap_smo_0_1, xticklabels='', yticklabels='',cmap = 'jet')
    # seaborn.heatmap(clu_ratemap_smo, xticklabels='', yticklabels='',cmap = 'jet',center=mean_rate,vmin=0,vmax=2*mean_rate)
    # plt.xticks(index_tet,['cage1','cage2','cage3','cage4','social','unsocial'])#设置横坐标刻
    # plt.title(f'heat map {uts.quality} clu{cluid} {fn} {experiment_tag[ses_n]}')
    # plt.savefig(fpath_fig+f'\\{cluid}\\heat map {uts.quality} clu{cluid} {fn} {experiment_tag[ses_n]}')
    # plt.show()    
def scatter_map():
    pass# if label！=0 color=yellow
        # if label ==0 color = red

def ego_rate_map(fr_pos,zone, dwell, dwell_smo,mean_rate,binsize=2):
    fr_pos[fr_pos<0]=0
    
    fr_pos=(fr_pos/binsize).astype('int32')
    
    fr_pos[:,0][fr_pos[:,0]>=(dwell.shape[0]-1)]=dwell.shape[1]-1
    fr_pos[:,1][fr_pos[:,1]>=(dwell.shape[1]-1)]=dwell.shape[0]-1
    clu_map = np.flip(npg.aggregate(np.flip(fr_pos.T), np.ones((np.size(fr_pos,axis=0)), dtype=int), size = list(np.shape(dwell_smo))), axis=0)
    # kdx,kdy = np.ogrid[-2:3,-2:3]
    # gk = np.exp(-(kdx*kdx +kdy*kdy) / 2*1*1)
    # gk = gk/np.sum(gk)
    # # try different kernel, boxcar here.
    bck = np.ones((5,5))/(5*5)
    dwell_smo=np.where(dwell_smo==0,np.nan,dwell_smo)
    # clu_map_smo = signal.convolve(clu_map, gk, 'same', 'direct')
    # clu_map_smo = signal.convolve(clu_map, bck, 'same', 'direct')
    
    clu_map_smo = gaussian_filter(clu_map, sigma=1)
    clu_ratemap_smo = np.divide(clu_map_smo, dwell_smo) 
    clu_ratemap=np.divide(clu_map,dwell)
    # mask = np.zeros_like(clu_ratemap_smo)
    # mask[clu_ratemap_smo == np.nan] = 250
    mask = np.isnan(clu_ratemap_smo)

    # 将 mask 的值设置为白色 (1.0)
    clu_ratemap_smo[mask] = 1.0
    # clu_ratemap_smo[dwell_smo == np.nan]=np.nan
    # clu_ratemap= np.nan_to_num(clu_ratemap)
    # mean_rate=np.nanmean(clu_ratemap)    
    # max_rate=np.quantile(clu_ratemap, 0.75)
    
    # spainfo = np.nansum((dwell_smo/np.sum(dwell_smo)) * (clu_ratemap/meanfiringrate) * np.log2(clu_ratemap/meanfiringrate))
    # seaborn.heatmap(clu_ratemap, xticklabels='', yticklabels='',cmap = 'jet',center=mean_rate,vmin=0,vmax=2*mean_rate)
    # plt.show()
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(clu_ratemap_smo, mask=mask,xticklabels='', yticklabels='',cmap = 'jet',center=mean_rate,vmin=0,vmax=2*mean_rate)
    # seaborn.heatmap(clu_ratemap_smo, xticklabels='', yticklabels='',cmap = 'jet',center=mean_rate,vmin=0,vmax=2*mean_rate)
    # plt.xticks(index_tet,['cage1','cage2','cage3','cage4','social','unsocial'])#设置横坐标刻
    # plt.title(f'heat map {uts.quality} clu{cluid} {fn} {experiment_tag[ses_n]}')
    # plt.savefig(fpath_fig+f'\\{cluid}\\heat map {uts.quality} clu{cluid} {fn} {experiment_tag[ses_n]}')
    # plt.show()    
        
        

    
# -----------------------------------------------------------------------------
#                 Derived Classes from Session
# -----------------------------------------------------------------------------
# class DetourSession(Session):
#     def __init__(self, dlch5, dlc_col_ind_dict, frame_state, vsync, sync_rate, expriment_tag, ses_id=0, fontsize=15):
#         Session.__init__(self, dlch5, dlc_col_ind_dict, vsync, sync_rate, experiment_tag, ses_id, fontsize)
                
def generate_interpolater(self):
    time2frame = interpolate.interp1d(self.frametime, np.linspace(0, (np.size(self.frametime)-1),num=np.size(self.frametime)), kind='nearest')
    self.get = {'time2frame':time2frame }
    
    def time2index(spiketime):
        frame_id = self.get['time2frame'](spiketime)
        return int(frame_id)
    def generate_index_func(obj, key):
        def FUNC(spiketime):
            return getattr(obj, key)[time2index(spiketime)]
        return FUNC
    def generate_mergeXY_func(obj, key):
        def FUNC(spiketime):
            return [obj.get[key+'X'](spiketime),obj.get[key+'Y'](spiketime)]
        return FUNC
                
    for key in vars(self):
        FrameData = getattr(self, key)
        if hasattr(FrameData, 'shape') and FrameData.shape[0] == self.frame_length:
            if type(FrameData[0]) == np.bool_ :                             #如果是bool值，那么在进行插值的时候应该取最邻近的值，而且bool值可以被数字比大小，所以单独写一个分支
                 self.get[key] = generate_index_func(self, key) 
            elif type(FrameData[0]) == str :                                #如果是字符串，那么在进行插值的时候应该取最邻近的值
                self.get[key] = generate_index_func(self, key) 
            elif type(FrameData[0]) == np.ndarray :                         #如果是一个数组，那意味着选到了某个位置（x，y），所以分别对x，y做插值，输出一个（x，y）
                self.get[key+'X'] = interpolate.UnivariateSpline(self.frametime, FrameData[:,0])
                self.get[key+'Y'] = interpolate.UnivariateSpline(self.frametime, FrameData[:,1])
                self.get[key] = generate_mergeXY_func(self, key) 
            elif FrameData[0] >= 0:                                         #如果是一个数字
                self.get[key] = interpolate.UnivariateSpline(self.frametime, FrameData)
            else:
                raise Exception('Error in generating interpolater, value type not defined')
            
    
def slowget(self, key, spiketime):
    ValueSet = getattr(self, key)
        
    if type(ValueSet[0]) == np.bool_ :        #如果是bool值，那么在进行插值的时候应该取最邻近的值，而且bool值可以被数字比大小，所以单独写一个分支
        Interpolater = interpolate.interp1d(self.frametime, ValueSet, kind='previous')
        ValueAtTime = Interpolater(spiketime)
        return ValueAtTime
        
    elif type(ValueSet[0]) == str :           #如果是字符串，那么在进行插值的时候应该取最邻近的值
        Interpolater = interpolate.UnivariateSpline(self.frametime, ValueSet)
        ValueAtTime = Interpolater(spiketime)
        return ValueAtTime
        
    elif type(ValueSet[0]) == np.ndarray :    #如果是一个数组，那意味着选到了某个位置（x，y），所以分别对x，y做插值，输出一个（x，y）
        InterpolaterX = interpolate.UnivariateSpline(self.frametime, ValueSet[:,0])
        InterpolaterY = interpolate.UnivariateSpline(self.frametime, ValueSet[:,1])
        ValueAtTime = [InterpolaterX(spiketime),InterpolaterY(spiketime)]
        return ValueAtTime
        
    elif ValueSet[0] >=0 :              #如果是一个数字
        Interpolater = interpolate.UnivariateSpline(self.frametime, ValueSet)
        ValueAtTime = Interpolater(spiketime)
        return ValueAtTime        
    else:
        raise Exception('You acquired wrong variable')


        

 # ----------------------------------------------------------------------------
 #                  Classes unit
 # ----------------------------------------------------------------------------
 
class Unit(object):
    def __init__(self, cluid, spike_pack, quality, Nses, experiment_tag, fontsize):
        self.cluid = cluid
        # self.quality = quality
        self.type = 'unknown'#IN or PC  
        self.meanwaveform = []
        self.Nses = Nses
        self.fontsize = fontsize
        self.experiment_tag = experiment_tag
        self.timestamps = [0 for i in range(Nses)]
        self.spiketime = [0 for i in range(Nses)]
        self.df=[0 for i in range(Nses)]
        self.mean_rate=[0 for i in range(Nses)]
        for i in range(Nses):
            spike_pick_cluid = np.isin(spike_pack[i][0],self.cluid)
            # spike_stay_pick_cluid = np.where(spike_pack[i][4] == self.cluid)[0]
            self.timestamps[i] = spike_pack[i][1][spike_pick_cluid]
            self.spiketime[i] = spike_pack[i][2][spike_pick_cluid]
            self.df[i]=pd.DataFrame({'spiketime':self.spiketime[i],'cluid':spike_pack[i][0][spike_pick_cluid]},columns=['spiketime','cluid'])
            try:
                self.mean_rate[i]=(self.df[i]['cluid'].value_counts())/(self.spiketime[i].max())
            except:
                self.mean_rate[i]=0/(self.spiketime[i].max())
        
        df=pd.concat(self.mean_rate,axis=1)
        self.mean_rate=df.mean(axis=1)
    def simple_putative_IN_PC_by_firingrate(self, ses):
        # this is a simple way to put IN and PC not by waveform but only global mean firing rate.
        # see Nuenuebel 2013, DR with L/MEC, threshold of mean firing rate is 10Hz.
        ses_inds = [i.id for i in ses]
        if (np.array(self.global_mean_rate)[ses_inds].all() > 10).all() == True:
            self.type = 'inhibitory'
        elif (np.array(self.global_mean_rate)[ses_inds] < 10).all() == True:
            self.type = 'excitatory'
        else:
            self.type = 'unsure' 
        
            
    def report_spatial(self):
        if 'spatial' in self.experiment_tag:
            print('cluster id:', self.cluid, '\n Nspike:', self.Nspikes_total, '\n peakrate:', self.peakrate, '\n mean rate while running:', self.__running_mean_rate, '\n spa. info:', self.spatial_info, '\n stability:', self.stability)
        else:
            print('you might used wrong method.')
    
    
    def raster_plot_peri_stimulus(self, ses, signal_on_time, pre_sec=30, post_sec=30, stim_color='yellow'):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_title('PSTH around laser-on, clu'+str(self.cluid), fontsize=self.fontsize*1.3)
        ax.set_xlabel('Time in sec', fontsize=self.fontsize)
        ax.set_xlim(left=-(pre_sec), right=post_sec)
        ax.set_ylabel('Trials', fontsize=self.fontsize)
        
        if self.Nses == 1:
            signal_on_temp = signal_on_time
        else:
            # signal_on_temp = signal_on_time[ses.id]
            signal_on_temp = signal_on_time
        ax.set_ylim(bottom=1, top=np.size(signal_on_temp)+1)
        if 'spatial' in self.experiment_tag:
            spike_time = np.concatenate((self.spiketime[ses.id], self.spiketime_stay[ses.id]))
        else:
            spike_time = self.spike_time
        for i in range(np.size(signal_on_temp)):
            spike_time_temp = spike_time[np.where(spike_time < (signal_on_temp[i] + post_sec))]
            spike_time_temp = spike_time_temp[np.where(spike_time_temp > (signal_on_temp[i] - pre_sec))].astype('float64')
            spike_time_temp = spike_time_temp - signal_on_temp[i]
            ax.scatter(spike_time_temp, np.array([i+1]*np.size(spike_time_temp)).astype('uint16'), c='k', marker='|', s=40)
        ax.fill_between([0, post_sec], 0, np.size(signal_on_temp)+1, facecolor=stim_color, alpha=0.5)
   
         
    def opto_inhibitory_tagging(self, ses, signal_on_time, mode, p_threshold=0.01, laser_on_sec=30, laser_off_sec=30, shuffle_range_sec=30):
        # it takes laser on as start of a cycle.
        if self.Nses == 1:
            signal_on_temp = signal_on_time
        else:
            signal_on_temp = signal_on_time[ses.id]
        if 'spatial' in self.experiment_tag:
            spike_time = np.concatenate((self.spiketime[ses.id], self.spiketime_stay[ses.id]))
        else:
            spike_time = self.spiketime
        if mode == 'ranksum':# in ranksum test mode, shuffle range is not used.
            on_spk_count = []
            off_spk_count = []
            for i in range(np.size(signal_on_temp)):
                spike_time_temp = spike_time[np.where(spike_time < (signal_on_temp[i] + laser_on_sec))]
                spike_time_temp = spike_time_temp[np.where(spike_time_temp > signal_on_temp[i])]
                on_spk_count.append(np.size(spike_time_temp))
                spike_time_temp2 = spike_time[np.where(spike_time < (signal_on_temp[i] + (laser_on_sec+laser_off_sec)))]
                spike_time_temp2 = spike_time_temp2[np.where(spike_time_temp2 > (signal_on_temp[i] + laser_on_sec))]
                off_spk_count.append(np.size(spike_time_temp2))
            statistic, pvalue = stats.ranksums(on_spk_count, off_spk_count, alternative='less')
            if pvalue < p_threshold:
                print('clu{} is positive, p-value'.format(self.cluid), pvalue)
                self.opto_tag = 'positive'
            # emmm...how to deal with margitial?
            else:
                print('clu{} is negative, p-value'.format(self.cluid), pvalue)
                self.opto_tag = 'negative'
        elif mode == 'shuffle':
            raise Exception('shuffle test is not finished yet')
            # 1000times, 0.01
            pass            
        else:
            raise Exception('Wrong mode or the mode has not been coded.')
    def get_mean_waveforms(self, waveforms):
        #to properly use this, I extract waveforms from phy2 with 'extract-waveforms' in command prompt.
        # and, the __init__.py is modified with max_nspikes=2000, nchannels=4, though it turns out 12chs still.
     
        unit_KSchannels = waveforms['channel'][np.where(waveforms['cluster'] == self.KScluid)][0]
        unit_waveforms_16ch = waveforms['raw'][np.where(waveforms['cluster'] == self.KScluid)]
        self.waveforms = np.zeros([unit_waveforms_16ch.shape[0], 60, 4])
        for ich in range(4):
            self.waveforms[:,:,ich] = unit_waveforms_16ch[:,:,np.where(unit_KSchannels == (self.tetrode*4+ich))[0][0]]
        
        self.mean_waveforms = np.mean(self.waveforms, axis=0)
    def plot_PSTH(self, ses):
        pass
    '''
    改！！！！！！！！！！
    '''
    # def get_mean_waveforms(self, waveforms):
    # #to properly use this, I extract waveforms from phy2 with 'extract-waveforms' in command prompt.
    # # and, the __init__.py is modified with max_nspikes=2000, nchannels=4, though it turns out 12chs still.
 
    # unit_KSchannels = waveforms['channel'][np.where(waveforms['cluster'] == self.KScluid)][0]
    # unit_waveforms_16ch = waveforms['raw'][np.where(waveforms['cluster'] == self.KScluid)]
    # self.waveforms = np.zeros([unit_waveforms_16ch.shape[0], 60, 4])
    # for ich in range(4):
    #     self.waveforms[:,:,ich] = unit_waveforms_16ch[:,:,np.where(unit_KSchannels == (self.tetrode*4+ich))[0][0]]
    
    # self.mean_waveforms = np.mean(self.waveforms, axis=0)
class Unit_social(Unit):
    def PSTH_signal(self):
        pass
    




        
        
