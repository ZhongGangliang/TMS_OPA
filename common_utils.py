# -*- coding=utf-8 -*-
import numpy as np
import csv
import pandas as pd
import tkinter
import math
import os
import progressbar
import hashlib
import pickle
from tkinter import filedialog
from cal_area import tarAreaNums
from skip_index import skipIndexs

def load_csv_as_dict(filename, row_indexs=None):
    csv_ = pd.read_csv(filename, index_col=False)
    if row_indexs is not None:
        sub_csv_ = csv_.iloc[row_indexs]
    else:
        sub_csv_ = csv_
    return sub_csv_


def merge_csv(filename_lst, row_indexs=None):
    merge_csv = pd.DataFrame()
    for filename in filename_lst:
        sub_csv = load_csv_as_dict(filename, row_indexs)
        merge_csv = merge_csv.append(sub_csv, ignore_index=True)
    return merge_csv


def depart_csv(data, group_by='index'):
    return data.groupby(group_by)


def save_csv(csv_data, file_name):
    csv_data.to_csv(file_name)

def get_file_md5(files):
    ''''
    计算文件的md5
    '''
    m = hashlib.md5()
    for file_name in files:
        with open(file_name,'rb') as fobj:
            while True:
                data = fobj.read(4096)
                if not data:
                    break
                m.update(data)
    m.update(str(skipIndexs).encode('utf-8'))
    return m.hexdigest()

def drop_index(all_csv, indexs):
    for idx in indexs:
        all_csv = all_csv.drop(all_csv[all_csv['index']==idx].index)
    return all_csv

def get_simu_result(subject, simu_result_path):
    '''
    合并csv表格
    '''
    from analysis_config import dataPath
    lastOptPath = dataPath + str(subject) + '/' + str(simu_result_path) + '/'
    fnames = []
    for path, _, filenames in os.walk(lastOptPath):
        for file in filenames:
            if file.endswith('.csv') and file.startswith('optimalResults_'):
                fnames.append(path + file)
    if len(fnames) == 0:
        print('There is no csv file in ' + lastOptPath)
        return None
    file_hash = lastOptPath + get_file_md5(fnames) + 'pkl'
    # 检查文件是否存在
    if os.path.exists(file_hash):
        all_csv = pickle.load(open(file_hash, 'rb'))
    else:
        # 读取所有csv文件
        all_csv = merge_csv(fnames)
        all_csv = drop_index(all_csv, skipIndexs)
        pickle.dump(all_csv, open(file_hash, 'wb'))
    return all_csv

def cal_angles(all_csv):
    '''
    计算每个刺激点的角度
    '''
    # 找到最大值
    all_csv['angle'] = 0
    # group_data = depart_csv(all_csv)
    
    
    first_all_csv = all_csv.drop_duplicates('index', keep='first') # 只保留第一个点的
    # assert (all_csv.shape[0] % first_all_csv.shape[0] == 0), "个别位点角度数量不一致"
    angles_number = all_csv['index'].value_counts()
    most_angle = np.argmax(np.bincount(angles_number))
    
    # # 解决TMS-atlas位置的最终绘制结果有中心空点的问题，，是因为绘制角度雷达图时删除了一些点。
    # # 绘制雷达图时需要保留tms_evaluation的6行和以下4行，不绘制雷达图时注释掉以下4行以及tms_evaluation的6行。
    # # 但是，目前发现是，绘制雷达图，删去这些点，才是正确的用于被试分析的数据集。不删点，虽然得到的atlas没有空心点，但是用于初试分析的数据集不对。
    
    # # 数据分析时：
    #            删点，保留——tms_evaluation 6行(绘制雷达图)以及common_utils的4行（删去resume计算重复点）。
    # # 显示mergeAtlas时：
    #            不删点，注释——tms_evaluation 6行(绘制雷达图)以及common_utils的4行（删去resume计算重复点）。
    # # 交叉留一计算时：
    #            删点，由于雷达图绘制错误故，注释——tms_evaluation 6行(绘制雷达图)，保留common_utils 4行（删去resume计算重复点）。
    # fail_index = angles_number[angles_number!=most_angle].index
    # if len(fail_index) > 0:
    #     all_csv = all_csv.drop(all_csv[all_csv['index'].isin(fail_index)].index)
    #     first_all_csv = all_csv.drop_duplicates('index', keep='first')

    if all_csv.shape[0] % first_all_csv.shape[0] == 0:
    
        coil_center = np.array(first_all_csv[['coil_center_x', 'coil_center_y', 'coil_center_z']])
        first_direction = np.array(first_all_csv[['coil_direction_x', 'coil_direction_y', 'coil_direction_z']]) - coil_center
        # 并重复
        first_direction = np.repeat(first_direction, int(all_csv.shape[0] / first_all_csv.shape[0]), axis=0)
        coil_center = np.repeat(coil_center, int(all_csv.shape[0] / first_all_csv.shape[0]), axis=0)

        # 计算与第一个位置之间的夹角
        direction = np.array(all_csv[['coil_direction_x', 'coil_direction_y', 'coil_direction_z']]) - coil_center
        degree_cos = (direction[:,0] * first_direction[:,0] + direction[:,1] * first_direction[:, 1] + direction[:,2] * first_direction[:,2]) / (np.linalg.norm(direction, axis=1) * np.linalg.norm(first_direction, axis=1))
        degree_cos[np.where(degree_cos > 1.0)[0]] = 1.0
        degree_cos[np.where(degree_cos < -1.0)[0]] = -1.0 
        degree = np.round(np.arccos(degree_cos) * 180.0 / math.pi)
        degree[degree==180.0] = 0
        all_csv['angle'] = degree
    else:
        print("WARNING: 个别位点角度数量不一致")
        group_data = depart_csv(all_csv)
        bar = progressbar.ProgressBar(0, len(group_data), redirect_stdout=True)
        count = 0
        for name, group in group_data:
            # 计算第一个位置
            coil_center = np.array(group[['coil_center_x', 'coil_center_y', 'coil_center_z']].iloc[0])
            first_direction = np.array(group[['coil_direction_x', 'coil_direction_y', 'coil_direction_z']].iloc[0]) - coil_center
            # 计算与第一个位置之间的夹角
            direction = np.array(group[['coil_direction_x', 'coil_direction_y', 'coil_direction_z']]) - coil_center   
            degree_cos = (direction[:,0] * first_direction[0] + direction[:,1] * first_direction[1] + direction[:,2] * first_direction[2]) / (np.linalg.norm(direction, axis=1) * np.linalg.norm(first_direction))
            degree_cos[np.where(degree_cos > 1.0)[0]] = 1.0
            degree_cos[np.where(degree_cos < -1.0)[0]] = -1.0 
            degree = np.round(np.arccos(degree_cos) * 180.0 / math.pi)
            degree[degree==180.0] = 0
            all_csv.loc[all_csv['index']==name, 'angle'] = degree
            # print(all_csv[all_csv['index']==name])
            # group['angle'] = degree
            bar.update(count)
            count = count + 1
        bar.finish()
        all_csv = all_csv.drop_duplicates(['index', 'angle'], keep='first')
        first_all_csv = all_csv.drop_duplicates('index', keep='first') # 只保留第一个点的
        angles_number = all_csv['index'].value_counts()
        most_angle = np.argmax(np.bincount(angles_number))
        fail_index = angles_number[angles_number!=most_angle].index
        if len(fail_index) > 0:
            all_csv = all_csv.drop(all_csv[all_csv['index'].isin(fail_index)].index)
            first_all_csv = all_csv.drop_duplicates('index', keep='first')
    return all_csv

def get_view_csv(all_csv):
    '''
    精简表格
    '''
    small_csv = pd.DataFrame()
    col_names = ['index', 'coil_center_x', 'coil_center_y', 'coil_center_z',
    'coil_direction_x', 'coil_direction_y','coil_direction_z',
    'coil_n_direction_x', 'coil_n_direction_y', 'coil_n_direction_z']
    for col in col_names:
        small_csv[col] = all_csv[col]
    for area_index in tarAreaNums:
        small_csv['value_'+str(area_index)] = all_csv['value_'+str(area_index)]
    return small_csv

def get_small_csv(all_csv):
    '''
    精简表格
    '''
    small_csv = pd.DataFrame()
    col_names = ['index', 'angle']
    for col in col_names:
        small_csv[col] = all_csv[col]
    for area_index in tarAreaNums:
        small_csv['value_'+str(area_index)] = all_csv['value_'+str(area_index)]
    return small_csv

def cal_func_value(all_csv, func):
    '''
    根据func计算目标函数
    params:
        all_csv: 所有表格
        func 目标函数func(all_csv, area_index)
    '''
    for area_index in tarAreaNums:
        all_csv['value_'+str(area_index)] = func(all_csv, area_index)
    return all_csv

def get_normalized(data):

    '''
    将data归一化
    算法:
        data_normalized = (data-min)/(max-min)
    '''
    if len(data)==0:
        return data
    min_data = np.array(data).min()
    max_data = np.array(data).max()
    if max_data == min_data:
        data_normalized = np.zeros(data.shape)
    else:
        data_normalized = (data-min_data)/(max_data-min_data)
    return data_normalized

def get_standardized(data):

    '''
    将data标准化（计算zscore）
    算法:
        data_standardized = (data-avg)/std
    '''
    avg_data = np.mean(data)
    std_data = np.std(data)
    data_standardized = (data-avg_data)/std_data
    return data_standardized

if __name__ == "__main__":
    func = lambda all_csv, area_index : 3.0 / all_csv['resultA_' + str(area_index)] + 1.0 / all_csv['resultB_B2/B1_' + str(area_index)]
    all_csv = get_simu_result()
    all_csv = cal_angles(all_csv)
    all_csv = cal_func_value(all_csv, func)
    all_csv = get_small_csv(all_csv)
    print(all_csv)
    # all_csv.to_csv('aaa.csv')