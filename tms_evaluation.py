# -*- coding=utf-8 -*-
import sys
from common_utils import *
from cal_area import tarAreaNums
import pandas as pd
import math
import scipy.signal
import numpy as np
import progressbar
import copy
import scipy.cluster.vq
import scipy.stats
import argparse
import matplotlib.pyplot as plt

def cal_better_points(all_csv, percentile):
    '''
    计算优选刺激位点
    params:
        all_csv: 已经计算完目标值的所有表格
        percentile: 百分位数，例如80为80%位数
    '''
    better_point_collections = {}
    # print('计算优选刺激位点')
    # bar = progressbar.ProgressBar(0, len(tarAreaNums), redirect_stdout=True)
    # count = 0
    for area_index in tarAreaNums:
        sort_ = all_csv.sort_values(by='value_' + str(area_index), ascending=False)
        rows = math.floor(sort_.shape[0] * (1.0 - percentile) / 1.0)
        better_point_collections[area_index] = sort_.head(rows)[['index', 'angle', 'value_' + str(area_index)]]
        # count = count + 1
        # bar.update(count)
    # bar.finish()
    return better_point_collections

def get_individual_angles_betterpoints(all_csv, better_point_collections_witout_angles, output):
    # for (k,v) in better_point_collections_witout_angles.items():
    #     better_points = all_csv[all_csv['index'].isin(np.array(v['index']))]
    #     group_data = depart_csv(better_points)
    #     fig = None
    #     ax = None
    #     index = 0
    #     for name, group in group_data:
    #         fig, ax = draw_radar(fig, ax, np.array(group['angle']), np.array(group['value_' + str(k)]), index)
    #         index = index + 1
    #     fig.savefig(output + '_' + str(k) + '_angle_radar.svg', dpi=600)
    for v in better_point_collections_witout_angles:
        better_points = all_csv[all_csv['index'].isin(np.array(v['position']))].copy()
        new_analysis_func = lambda data_csv, area_index : get_normalized(data_csv['value_' + str(area_index)])
        better_points = cal_func_value(better_points, new_analysis_func)
        group_data = depart_csv(better_points)
        fig = None
        ax = None
        index = 0
        names = []
        values = []
        for name, group in group_data:
            if np.asarray(group['angle'])[-1] == 180.0:
                name = np.array(group['angle'])[:-1]
                value = np.array(group['value_' + str(v['area_index'])])[:-1]
            else:
                name = np.array(group['angle'])
                value = np.array(group['value_' + str(v['area_index'])])
            values.append(value)
            names = np.copy(name)
            name = np.concatenate((name, name + 180.0))
            value = np.concatenate((value, value))
            index = index + 1
        values = np.array(values)
        if len(values.shape) != 2:
            print('每个个体计算角度不统一 无法生成统计图')
            continue
        value_avg = np.average(values.T, axis=1)
        value_std = np.std(values.T, axis=1)
        value_up = scipy.stats.norm.interval(0.95, loc=value_avg, scale=value_std / np.sqrt(len(values)))[1]
        value_down = scipy.stats.norm.interval(0.95, loc=value_avg, scale=value_std / np.sqrt(len(values)))[0]
        fig = None
        ax = None
        index = 0
        # names = []
        values = []
        
        names = np.concatenate((names, names + 180.0))
        value_avg = np.concatenate((value_avg, value_avg))
        value_up = np.concatenate((value_up, value_up))
        value_down = np.concatenate((value_down, value_down))
        index = index + 1
    return fig


def cal_better_points_oneangle(all_csv, percentile):
    '''
    计算优选刺激位点, 每个位置仅保留最大值的方向
    params:
        all_csv: 已经计算完目标值的所有表格
        percentile: 百分位数，例如80为80%位数
    '''
    group_data = depart_csv(all_csv)
    better_point_collections = {}
    # print('计算优选刺激位点')
    # bar = progressbar.ProgressBar(0, len(tarAreaNums), redirect_stdout=True)
    # count = 0
    for area_index in tarAreaNums:
        sort_ = all_csv.sort_values(by='value_' + str(area_index), ascending=False)
        oneangle_all_csv = sort_.drop_duplicates('index', keep='first')
        sort_ = oneangle_all_csv
        rows = math.floor(sort_.shape[0] * (1.0 - percentile) / 1.0)
        better_point_collections[area_index] = sort_.head(rows)[['index', 'angle', 'value_' + str(area_index)]]
        # count = count + 1
        # bar.update(count)
    # bar.finish()
    return better_point_collections

def cal_focality(group_better_point_collections, first_p, second_p):
    '''
    计算集中度
    params:
        better_point_collections: 优选集
        first_p: 概率1
        second_p: 概率2
    '''
    focalities = []
    # print('计算集中度')
    # bar = progressbar.ProgressBar(0, len(group_better_point_collections), redirect_stdout=True)
    # count = 0
    # 群组
    for area_index, better_point_collections in group_better_point_collections.items():
        # 每个index出现的次数
        index_probability = None
        # 个体
        for collections_dict in better_point_collections:
            subject_better_point_collections = collections_dict['collections']
            subject = collections_dict['subject']
            temp_index_probability = pd.Series(data=1.0, index=subject_better_point_collections['index'].unique())
            if index_probability is None:
                index_probability = temp_index_probability
            else:
                index_probability = index_probability.add(temp_index_probability, fill_value=0)
        
        index_probability = index_probability / len(better_point_collections)
        bigger_than_first_p = (index_probability > first_p).sum()
        bigger_than_second_p = (index_probability > second_p).sum()
        focalities.append(
                {
                    'area_index': area_index,
                    'focality': bigger_than_first_p / bigger_than_second_p
                }
            )
    #     count = count + 1
    #     bar.update(count)
    # bar.finish()
    return focalities

def save_group_pos_select(index_x, index_y, index_angle, value, brain_area_index, filename):
    a = (np.array(index_x) + start_index_i) / 101.0 * 180.0
    b = (np.array(index_y) + start_index_j) / 101.0 * 180.0
    c = None
    if len(index_angle) == 1:
        c = np.array([index_angle]*len(a))
    else:
        c = np.array(index_angle)
    d = np.array(value)
    e = None
    if len(brain_area_index) == 1:
        e = np.array([brain_area_index]*len(a))
    else:
        e = np.array(brain_area_index)

    a=a.reshape((len(a),1))
    b=b.reshape((len(b),1))
    c=c.reshape((len(c),1))
    d=d.reshape((len(d),1))
    e=e.reshape((len(e),1))

    f=np.concatenate((a,b,c,d,e),axis=1)

    df = pd.DataFrame(f)

    df["index_x"] = df.pop(0)
    df["index_y"] = df.pop(1)
    df["angle"] = df.pop(2)
    df["value"] = df.pop(3)
    df["barin_area_index"] = df.pop(4)

    df.to_csv(filename)
    


def cal_group_pos_select(group_better_point_collections, threshold_p, output_path, save_fig=True):
    '''
    计算群组位置
    params:
        better_point_collections: 优选集
        threshold_p: 概率
    '''
    # 统计pointmap中每个point的值的分布情况
    value_distribution_pointmap = {}
    positions = []
    # print('计算群组位置')
    # bar = progressbar.ProgressBar(0, len(group_better_point_collections), redirect_stdout=True)
    # count = 0
    # 群组
    for area_index, better_point_collections in group_better_point_collections.items():
        # 每个index出现的次数
        index_probability = None
        value_sum = None
        angle_sum = None
        # 个体
        for collections_dict in better_point_collections:
            subject_better_point_collections = collections_dict['collections']
            subject = collections_dict['subject']
            temp_index_probability = pd.Series(data=1.0, index=subject_better_point_collections['index'].unique())
            temp_value_sum = pd.Series(data=subject_better_point_collections['value_'+str(area_index)].values, index=subject_better_point_collections['index'].unique())
            temp_angle_sum = pd.Series(data=subject_better_point_collections['angle'].values, index=subject_better_point_collections['index'].unique())

            if index_probability is None:
                index_probability = temp_index_probability
            else:
                index_probability = index_probability.add(temp_index_probability, fill_value=0)
            
            if value_sum is None:
                value_sum = temp_value_sum
            else:
                value_sum = value_sum.add(temp_value_sum, fill_value=0)

            if angle_sum is None:
                angle_sum = temp_angle_sum
            else:
                angle_sum = angle_sum.add(temp_angle_sum, fill_value=0)

        value_sum = value_sum.div(index_probability, fill_value=1)
        value_std = None

        angle_sum = angle_sum.div(index_probability, fill_value=1)
        angle_std = None

        # 个体
        for collections_dict in better_point_collections:
            subject_better_point_collections = collections_dict['collections']
            subject = collections_dict['subject']

            temp_value_std = pd.Series(data=subject_better_point_collections['value_'+str(area_index)].values, index=subject_better_point_collections['index'].unique())
            temp_value_std = temp_value_std.sub(value_sum, fill_value=0)
            temp_value_std = temp_value_std.mul(temp_value_std, fill_value=0)

            temp_angle_std = pd.Series(data=subject_better_point_collections['angle'].values, index=subject_better_point_collections['index'].unique())
            temp_angle_std = temp_angle_std.sub(angle_sum, fill_value=0)
            temp_angle_std = temp_angle_std.mul(temp_angle_std, fill_value=0) 

            if value_std is None:
                value_std = temp_value_std
            else:
                # value_std_sqrt = value_std.add(temp_value_std.sub(value_sum, fill_value=0), fill_value=0)
                # value_std = value_std_sqrt.mul(value_std_sqrt, fill_value=0)
                value_std = value_std.add(temp_value_std, fill_value=0)

            if angle_std is None:
                angle_std = temp_angle_std
            else:
                # angle_std_sqrt = angle_std.add(temp_angle_std.sub(angle_sum, fill_value=0), fill_value=0)
                # angle_std = angle_std_sqrt.mul(angle_std_sqrt, fill_value=0)
                angle_std = angle_std.add(temp_angle_std, fill_value=0)

        value_std = value_std.div(index_probability, fill_value=1)
        angle_std = angle_std.div(index_probability, fill_value=1)

        index_probability = index_probability / len(better_point_collections)
        bigger_than_threshold_indexes = index_probability[index_probability > threshold_p].values * area_index
        bigger_than_threshold_values = value_sum[index_probability > threshold_p].values
        bigger_than_threshold_angle = angle_sum[index_probability > threshold_p].values

        bigger_than_threshold_values_std = value_std[index_probability > threshold_p].values
        bigger_than_threshold_angle_std = angle_std[index_probability > threshold_p].values

        bigger_than_threshold_values_std = np.sqrt(bigger_than_threshold_values_std)
        bigger_than_threshold_angle_std = np.sqrt(bigger_than_threshold_angle_std)

        bigger_than_threshold_p = index_probability[index_probability > threshold_p].index
        bigger_than_threshold_p = bigger_than_threshold_p.values.astype(np.int32)

        unique_index_i, unique_index_j = np.unravel_index(bigger_than_threshold_p, (101, 101))
        unique_index_i = unique_index_i - start_index_i
        unique_index_j = unique_index_j - start_index_j

        # 重心
        mx = np.sum(unique_index_i * bigger_than_threshold_values) / np.sum(bigger_than_threshold_values)
        my = np.sum(unique_index_j * bigger_than_threshold_values) / np.sum(bigger_than_threshold_values)

        classify_values = classify_func(
            mx=mx, my=my, 
            xs=unique_index_i, 
            ys=unique_index_j,
            values=bigger_than_threshold_values,
            std_values=bigger_than_threshold_values_std,
            std_angles=bigger_than_threshold_angle_std
            )

        angle_center = np.sum(bigger_than_threshold_angle * bigger_than_threshold_values) / np.sum(bigger_than_threshold_values)

        positions.append(
                {
                    'area_index': area_index,
                    'position': bigger_than_threshold_p,
                    'points': classify_values,
                    'values': bigger_than_threshold_values,
                    'angles': bigger_than_threshold_angle,
                    'angle':angle_center,
                    'mx': mx,
                    'my':my
                }
            )  

        save_group_pos_select(unique_index_i, unique_index_j, [-1], bigger_than_threshold_indexes, [area_index], output_path + str(area_index) + '_' + str(threshold_p) + '_pos.csv')
        save_group_pos_select(unique_index_i, unique_index_j, [-1], bigger_than_threshold_values, [area_index], output_path + str(area_index) + '_' + str(threshold_p) + '_avg_values.csv')
        save_group_pos_select(unique_index_i, unique_index_j, [-1], bigger_than_threshold_angle, [area_index], output_path + str(area_index) + '_' + str(threshold_p) + '_avg_angle.csv')
        save_group_pos_select(unique_index_i, unique_index_j, [-1], bigger_than_threshold_values_std, [area_index], output_path + str(area_index) + '_' + str(threshold_p) + '_std_values.csv')
        save_group_pos_select(unique_index_i, unique_index_j, [-1], bigger_than_threshold_angle_std, [area_index], output_path + str(area_index) + '_' + str(threshold_p) + '_std_angle.csv')

    #     count = count + 1
    #     bar.update(count)
    # bar.finish()
    return positions

def cal_extre(data):
    # 左右扩充
    # 右边扩充最左边的值
    data_ = np.insert(data, len(data), data[0], axis=0)
    # 左边扩充最右边的值
    data_ = np.insert(data_, 0, data[-1], axis=0)
    # 左移
    data_left = np.insert(data_[1:], len(data_[1:]), 0, axis=0)
    origin_left = (data_ - data_left)[1:-1] # >0表示比右边的数字大
    # 0情况下取右边的值
    last_no_zero = 0
    for i in range(len(origin_left) - 1, -1, -1):
        if origin_left[i] != 0:
            last_no_zero = origin_left[i]
        else:
            origin_left[i] = last_no_zero
    # 右移
    data_right = np.insert(data_[:-1], 0, 0, axis=0)
    origin_right = (data_ - data_right)[1:-1] # >0表示比左边的数字大
    # 0情况下取左边的值
    last_no_zero = 0
    for i in range(len(origin_right)):
        if origin_right[i] != 0:
            last_no_zero = origin_right[i]
        else:
            origin_right[i] = last_no_zero
    # 比左右都大的即为极值
    extra_pos = (origin_left > 0) * (origin_right > 0)
    return extra_pos

def cal_direction_extre_num(group_better_point_collections, threshold_scale):
    '''
    计算方向极值数
    params:
        better_point_collections: 优选集
        threshold_scale: 最大值阈值比例
    '''
    group_extre = []
    individuation_extre = []
    # print('计算方向极值数')
    # bar = progressbar.ProgressBar(0, len(group_better_point_collections), redirect_stdout=True)
    # count = 0
    # 群组
    for area_index, better_point_collections in group_better_point_collections.items():
        # 每个angle出现的次数
        angle_probability = None
        angle_value_angle = np.array([])
        angle_value_value = np.array([])
        temp_extre_nums_individuation = {}
        temp_extre_angles_individuation = {}
        group_over_threshold_collections = pd.DataFrame()
        # 个体
        for collections_dict in better_point_collections:
            subject_better_point_collections = collections_dict['collections']
            subject = collections_dict['subject']
            value_max = subject_better_point_collections['value_' + str(area_index)].max()
            threshold = threshold_scale * value_max
            #  选取每个位点目标值最大的角度
            # index 重复部分
            index_groups_ = subject_better_point_collections.groupby('index')
            over_threshold_collections = pd.DataFrame()
            
            index_groups_ = subject_better_point_collections.sort_values(by='value_' + str(area_index), ascending=False)
            sort_ = index_groups_.drop_duplicates('index', keep='first')

            rows = math.floor(sort_.shape[0] * (1 - threshold_scale) / 1.0)
            over_threshold_collections = sort_.head(rows)
            temp_angle_probability = pd.Series(over_threshold_collections['angle']).value_counts()

            if angle_probability is None:
                angle_probability = temp_angle_probability
            else:
                angle_probability = angle_probability.add(temp_angle_probability, fill_value=0)
            angle_value_angle = np.append(angle_value_angle, over_threshold_collections['angle'])
            angle_value_value = np.append(angle_value_value, over_threshold_collections['value_' + str(area_index)])
            # 分析个体极值个数
            extre_angles_pos = cal_extre(np.array(temp_angle_probability))
            extre_angles = np.array(temp_angle_probability.index)[extre_angles_pos]
            extre_nums = np.array(temp_angle_probability)[extre_angles_pos]
            individuation_extre_elm = {}
            individuation_extre_elm['area_index'] = area_index
            individuation_extre_elm['subject'] = subject
            individuation_extre_elm['extre_angles'] = extre_angles
            individuation_extre_elm['extre_nums'] = extre_nums
            extre_values = []
            for ea in extre_angles:
                value = np.average(over_threshold_collections[over_threshold_collections['angle']==ea]['value_' + str(area_index)])
                extre_values.append(value)
            individuation_extre_elm['extre_values'] = extre_values
            individuation_extre.append(individuation_extre_elm)
        # 分析群组极值个数 
        extre_angles_pos = cal_extre(np.array(angle_probability))
        extre_angles = np.array(angle_probability.index)[extre_angles_pos]
        extre_nums = np.array(angle_probability)[extre_angles_pos]
        # bar.update(count)
        # count = count + 1
        group_extre_elm = {}
        group_extre_elm['area_index'] = area_index
        group_extre_elm['extre_angles'] = extre_angles
        group_extre_elm['extre_nums'] = extre_nums
        extre_values = []
        for ea in extre_angles:
            value = np.average(angle_value_value[np.where(angle_value_angle==ea)[0]])
            extre_values.append(value)
        group_extre_elm['extre_values'] = extre_values
        # for ea in extre_angles:
        #     value = np.average(over_threshold_collections[over_threshold_collections['angle']==ea]['value_' + str(area_index)])
        #     extre_values.append(value)
        # group_extre_elm['extre_values'] = extre_values
        group_extre.append(group_extre_elm)

    # bar.finish()
    return [group_extre, individuation_extre]

def normalize_individual_func_value(all_csv):
    from tms_atlas.cal_area import tarAreaNums
    from common_utils import get_normalized
    for area_index in tarAreaNums:
        all_csv['value_'+str(area_index)] = get_normalized(all_csv['value_'+str(area_index)])
    return all_csv

def standardize_individual_func_value(all_csv):
    from tms_atlas.cal_area import tarAreaNums
    from common_utils import get_standardized
    for area_index in tarAreaNums:
        all_csv['A_'+str(area_index)] = get_standardized(all_csv['A_'+str(area_index)])
        all_csv['B_B2/B1_'+str(area_index)] = get_standardized(all_csv['B_B2/B1_'+str(area_index)])
    return all_csv

def merge_pointmap(position, output_path):
    # 合并points
    points = np.array([])
    for area in position:
        points = np.append(points, area['points'])
    if len(points) == 0:
        return
    # 归一化合并后的points
    from common_utils import get_normalized
    points = get_normalized(points)
    # 还回去, 并统计每个pointmap所属的脑区
    points_count = 0
    pointmap_belong = {}
    pointmap_points = {}
    pointmap_values = {}
    pointmap_angles = {}
    pointmap_mx = {}
    pointmap_my = {}
    pointmap_angle = {}
    for area in position:
        area['points'] = points[points_count : points_count + len(area['points'])]
        for i, p in enumerate(area['position']):
            if p in pointmap_belong.keys():
                pointmap_belong[p].append(area['area_index'])
                pointmap_points[p].append(area['points'][i])
                pointmap_values[p].append(area['values'][i])
                pointmap_angles[p].append(area['angles'][i])
                # pointmap_mx[p].append(area['mx'])
                # pointmap_my[p].append(area['my'])
                # pointmap_angle[p].append(area['angle'])
            else:
                pointmap_belong[p] = [area['area_index']]
                pointmap_points[p] = [area['points'][i]]
                pointmap_values[p] = [area['values'][i]]
                pointmap_angles[p] = [area['angles'][i]]
                # pointmap_mx[p] = [area['mx']]
                # pointmap_my[p] = [area['my']]
                # pointmap_angle[p] = [area['angle']]
            # pointmap_belong[p] = pointmap_belong.get(p, []).append(area['area_index'])
            # pointmap_points[p] = pointmap_points.get(p, []).append(area['points'][i])
        points_count = points_count + len(area['points'])
    # merge每个pointmap所属的脑区
    unique_index = []
    unique_value = []
    unique_real_value = []
    unique_angle = []
    
    for p, area in pointmap_belong.items():
        real_values = pointmap_values[p]
        argmin = np.array(pointmap_points[p]).argmin()
        area_index = area[argmin]
        unique_index.append(p)
        unique_value.append(area_index)
        unique_real_value.append(real_values[argmin])
        unique_angle.append(pointmap_angles[p][argmin])
    unique_index_i, unique_index_j = np.unravel_index(unique_index, (101, 101))
    unique_index_i = unique_index_i - start_index_i
    unique_index_j = unique_index_j - start_index_j
    save_group_pos_select(unique_index_i, unique_index_j, [-1], unique_value, [-1], output_path + 'merge.csv')
    save_group_pos_select(unique_index_i, unique_index_j, unique_angle, unique_real_value, unique_value, output_path + 'tmsAtlas.csv')

def cal_tms_evalution(output_path, subjectnum_result, save_pickle=True, save_fig=True):
    if len(subjectnum_result) == 0:
        return None
    group_better_point_collections = {}
    group_better_point_collections_without_angles = {}
    for area_index in tarAreaNums:
        group_better_point_collections[area_index] = []
        group_better_point_collections_without_angles[area_index] = []
    for res in subjectnum_result:
        all_csv = get_simu_result(res['subject'], res['simu_result_path'])
        # all_csv = standardize_individual_func_value(all_csv)
        # 检测某个被试的位点或方向没有计算完，报错的被试即为有问题的被试
        print('subject: {},simuresult: {}'.format(res['subject'],res['simu_result_path']))
        all_csv = all_csv.drop_duplicates(['index', 'coil_direction_x', 'coil_direction_y', 'coil_direction_z'],keep='first')
        all_csv = cal_angles(all_csv)
        all_csv = cal_func_value(all_csv, analysis_func)
        all_csv = get_small_csv(all_csv)
        # all_csv = drop_index(all_csv, skipIndexs)

        # all_csv = normalize_individual_func_value(all_csv)
        better_point_collections = cal_better_points(all_csv, better_point_collections_percent)
        better_point_collections_witout_angles = cal_better_points_oneangle(all_csv, better_point_collections_percent)
        # fig = get_individual_angles_betterpoints(all_csv, better_point_collections_witout_angles, output_path + res['subject'])
        for key in better_point_collections.keys():
            group_better_point_collections[key].append(
                {'subject': res['subject'],
                 'collections': better_point_collections[key]})
            group_better_point_collections_without_angles[key].append(
                {'subject': res['subject'],
                 'collections': better_point_collections_witout_angles[key]})
    focalities = cal_focality(group_better_point_collections, focality_best_percent, focality_base_percent)
    positions = cal_group_pos_select(group_better_point_collections_without_angles, group_threshold, output_path, save_fig=save_fig)
    
    
    # import pickle
    # positions = pickle.load(open('position.pkl', 'rb'))
    merge_pointmap(positions, output_path)
    group_extre, individuation_extre = cal_direction_extre_num(group_better_point_collections, extre_angles_filter_percent)
    group_extre_df = pd.DataFrame(group_extre)
    individuation_extre_df = pd.DataFrame(individuation_extre)
    focalities_df = pd.DataFrame(focalities)
    group_extre_df.to_csv(output_path + 'group_extre.csv')
    individuation_extre_df.to_csv(output_path + 'individuation_extre.csv')
    focalities_df.to_csv(output_path + 'focalities.csv')
    final_ = {}
    result_table = pd.DataFrame()
    for area in positions:
        if len(area['position']) == 0:
            continue
        # 求重心位置
        unique_index_i, unique_index_j = np.unravel_index(area['position'], (101, 101))
        unique_value = area['values']
        mx = np.sum(unique_index_i * unique_value) / np.sum(unique_value)
        my = np.sum(unique_index_j * unique_value) / np.sum(unique_value)

        from tms_atlas.point_cloud_utils import getSUBPointmap

        res = {}
        res['area'] = area['area_index']
        res['eq_i'] = mx / 101.0 * 180.0
        res['lg_j'] = my / 101.0 * 180.0
        res['angle'] = area['angle']

        mnipoints = getSUBPointmap('MNI152')
        center, _ = mnipoints.cal_point(mx / 101.0 * 180.0, my /101.0 * 180.0)

        res['center_x'] = center[0]
        res['center_y'] = center[1]
        res['center_z'] = center[2]
        result_table = result_table.append(res, ignore_index=True)
    result_table.to_csv(output_path + 'result.csv')

    if save_pickle:
         import pickle
         pickle.dump(positions, open(output_path + 'position.pkl', 'wb'))
    return positions

def leave_one_out_cross_validation(output_path_all, cpus=8):
    '''
    交叉留一验证
    '''
    import multiprocessing
    import pickle
    processes = []
    # 留一计算部分
    pool = multiprocessing.Pool(processes=cpus)
    for res in subjectnum_result:
        output_path = output_path_all + 'leave_' + res['subject'] + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        subjectnum_result_temp = copy.copy(subjectnum_result)
        subjectnum_result_temp.remove(res)   
        process = pool.apply_async(func=cal_tms_evalution, args=(output_path, subjectnum_result_temp, True, False))
        processes.append(process)
        # processes.append(multiprocessing.Process(target=cal_tms_evalution, args=(output_path, subjectnum_result_temp, True)))
    output_path = output_path_all + 'leave_all/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    process = pool.apply_async(func=cal_tms_evalution, args=(output_path, subjectnum_result, True, True))
    processes.append(process)
    # processes.append(multiprocessing.Process(target=cal_tms_evalution, args=(output_path, subjectnum_result, True)))
    pool.close()
    print('循环留一计算开始')
    pool.join()
    # for p in processes:
    #     p.start()
    bar = progressbar.ProgressBar(0, len(processes), redirect_stdout=True)
    count = 0
    for p in processes:
        count = count + 1
        bar.update(count)
        p.get()
    bar.finish()
    # 交叉分析部分
    print("循环留一计算完成，开始交叉分析")
    pointmap = np.load(abPathPart() + "MNI_pointmap_final.npy")
    cross_table = pd.DataFrame()
    dice_table = pd.DataFrame()
    subjectnum_result.append({
            'subject': 'all',
            'simu_result_path': ''
        })
    output_path = output_path_all + 'leave_all' +  '/'
    if not os.path.exists(output_path):
        return
    positions_all = pickle.load(open(output_path + 'position.pkl', 'rb'))
    for res in subjectnum_result:
        output_path = output_path_all + 'leave_' + res['subject'] + '/'
        if not os.path.exists(output_path):
            break
        positions = pickle.load(open(output_path + 'position.pkl', 'rb'))
        cross_x = {}
        cross_y = {}
        cross_z = {}
        cross_angle = {}
        cross_dice = {}
        for area, area_all in zip(positions, positions_all):
            if len(area['position']) == 0:
                continue
            # 求重心位置
            unique_index_i, unique_index_j = np.unravel_index(area['position'], (101, 101))
            unique_value = area['values']
            mx = np.sum(unique_index_i * unique_value) / np.sum(unique_value)
            my = np.sum(unique_index_j * unique_value) / np.sum(unique_value)
            mx = int(round(mx))
            my = int(round(my))
            center = pointmap[mx, my, :]
            cross_x[area['area_index']] = center[0]
            cross_y[area['area_index']] = center[1]
            cross_z[area['area_index']] = center[2]
            cross_angle[area['area_index']] = area['angle']
            # 计算交并比
            if len(area_all['position']) == 0:
                continue
            leave_pos = np.array(area['position'])
            all_pos = np.array(area_all['position'])
            intersection = np.in1d(leave_pos, all_pos)
            union = np.in1d(leave_pos, all_pos)
            iou = len(intersection) / len(union)
            cross_dice[area['area_index']] = iou
        cross_table = pd.concat([cross_table, pd.DataFrame.from_dict(cross_x, orient='index', columns=['leave_' + res['subject'] + '_x'])], axis=1)
        cross_table = pd.concat([cross_table, pd.DataFrame.from_dict(cross_y, orient='index', columns=['leave_' + res['subject'] + '_y'])], axis=1)
        cross_table = pd.concat([cross_table, pd.DataFrame.from_dict(cross_z, orient='index', columns=['leave_' + res['subject'] + '_z'])], axis=1)
        cross_table = pd.concat([cross_table, pd.DataFrame.from_dict(cross_angle, orient='index', columns=['leave_' + res['subject'] + '_angle'])], axis=1)
        dice_table = pd.concat([dice_table, pd.DataFrame.from_dict(cross_dice, orient='index', columns=['leave_' + res['subject'] + '_dice'])], axis=1)
    from analysis_tools.nearestscalppoint import nearestscalppoint
    nearestscalp = nearestscalppoint()
    nearestscalp_d_x = {}
    nearestscalp_d_y = {}
    nearestscalp_d_z = {}
    for i, area in enumerate(tarAreaNums):
        if len(nearestscalp[i]) > 0:
            nearestscalp_d_x[area] = nearestscalp[i][0][0]
            nearestscalp_d_y[area] = nearestscalp[i][0][1]
            nearestscalp_d_z[area] = nearestscalp[i][0][2]
        else:
            nearestscalp_d_x[area] = math.nan
            nearestscalp_d_y[area] = math.nan
            nearestscalp_d_z[area] = math.nan
    cross_table = pd.concat([cross_table, pd.DataFrame.from_dict(nearestscalp_d_x, orient='index', columns=['nearest_x'])], axis=1)
    cross_table = pd.concat([cross_table, pd.DataFrame.from_dict(nearestscalp_d_y, orient='index', columns=['nearest_y'])], axis=1)
    cross_table = pd.concat([cross_table, pd.DataFrame.from_dict(nearestscalp_d_z, orient='index', columns=['nearest_z'])], axis=1)
    cross_table.to_csv(output_path_all + 'leaveOneOutCrossValidationResult.csv')
    dice_table.to_csv(output_path_all + 'leaveOneOutCrossDiceResult.csv')

def cal_iou(folder1, folder2):
    output_path = folder1
    if not os.path.exists(output_path):
        return
    positions1 = pickle.load(open(output_path + 'position.pkl', 'rb'))
    output_path2 = folder2
    if not os.path.exists(output_path2):
        return
    positions2 = pickle.load(open(output_path2 + 'position.pkl', 'rb'))
    cross_dice = {}
    dice_table = pd.DataFrame()
    for area1, area2 in zip(positions1, positions2):
        if len(area1['position']) == 0 or len(area2['position']) == 0:
            continue
        leave_pos = np.array(area1['position'])
        all_pos = np.array(area2['position'])
        intersection = np.in1d(leave_pos, all_pos)
        union = np.union1d(leave_pos, all_pos)
        iou = len(intersection) / len(union)
        cross_dice[area1['area_index']] = iou
    dice_table = pd.concat([dice_table, pd.DataFrame.from_dict(cross_dice, orient='index', columns=['dice'])], axis=1)
    dice_table.to_csv(folder1 + 'diceBetween' + folder2.replace('/', '_').replace(':','') + '.csv')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis TMS evaluation')
    parser.add_argument("-o", "--output", type=str,
                        help="Output folder")
    parser.add_argument("-x", "--cross", action="store_true",
                        help="Leave one out cross validation")
    parser.add_argument("-c", "--cpus", type=int,
                        help="cpus")
    parser.add_argument("-d", "--dice", type=str,
                        help="dice folder")
    args = vars(parser.parse_args())
    dataPath = '/DATA/TMS_DATA/'
    output_path = dataPath + 'analysis_path/' + args['output'] + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    from analysis_tools.analysis_config import *
    if args['cross']:
        # 预读取用于加速
        for res in subjectnum_result:
            all_csv = get_simu_result(res['subject'], res['simu_result_path'])
        leave_one_out_cross_validation(output_path, args["cpus"])
    elif args['dice']:
        output_path1 = dataPath + 'analysis_path/' + args['output'] + '/'
        output_path2 = dataPath + 'analysis_path/' + args['dice'] + '/'
        cal_iou(output_path1, output_path2)
    else:
        print('开始分析结果')
        cal_tms_evalution(output_path, subjectnum_result)
    # print(group_extre)
