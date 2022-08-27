subjectnum_result = [
        {
            'subject': '109123',
            'simu_result_path': 'simu_result20210830005924_125277'
        },
        {
            'subject': '198451',
            'simu_result_path': 'simu_result20201021092444_59355'
        }
    ]

dataPath = 'F:/Op_TMS_Coil_Pose_Atlas/NEVAdata/Atlas/'

# 优选点集百分比
better_point_collections_percent = 0.95
# 聚焦性指标分子百分比
focality_best_percent = 0.8
# 聚焦性指标分母百分比
focality_base_percent = 0.5
# 角度极值书指标选取百分比
extre_angles_filter_percent = 0.5

# 群组阈值
group_threshold = 0.95

start_index_i = 0
start_index_j = 51

# TEST 2 
from common_utils import get_standardized
analysis_func = lambda all_csv, area_index : get_standardized(1.0 / all_csv['resultA_' + str(area_index)]) + get_standardized(1.0 / all_csv['resultB_B2/B1_' + str(area_index)])


# 脑区分类的评价指标函数
def classify_func(mx, my, xs, ys, values, std_values, std_angles):
    import numpy as np
    # 距离重心距离
    distance = np.sqrt((xs - mx) * (xs - mx) + (ys - my) * (ys - my))
    # value倒数
    reciprocal_value = 1.0 / values
    func = distance + reciprocal_value + std_values + std_angles
    return func