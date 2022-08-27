# TMS_OPA: 添加一句话的介绍

## 0 介绍

添加介绍文字，提供以下功能：

- 功能1
- 功能2

## 1 安装与使用

1. 安装创建python环境，例如conda
2. 安装依赖库，请查看 requirements.txt

## 2 线圈位姿转换工具
### 2.1 功能
- 将XXX线圈坐标系转换为MNI坐标系下位姿


### 2.2 用法
```bash
cd ./TMS_OPA
python ./tms_convert.py -p {a} {b}
```
### 2.3 结果输出
- 线圈中心点所在位姿的MNI坐标

## 3 位姿图谱生成工具
### 3.1 功能
- 统计分析仿真计算结果，生成群组线圈位姿图谱
### 3.2 用法
#### 3.2.1 修改 analysis_config.py内容
- 仿真结果的subject与存放结果的目录
  - 修改文件`analysis_config.py`中相应的内容
  - 示例：
  ```python
  subjectnum_result = [
        {
            'subject': '151627', # 第一个subject
            'simu_result_path': 'simu_result20201226232724_21443' # 对应的仿真结果目录
        },
        {
            'subject': '160123',
            'simu_result_path': 'simu_result20210124015948_33635'
        }
    ]
  ```
- 评价指标参数
  - 修改文件`analysis_config.py`中相应的内容，具体参看注释
- 目标函数
  - 修改文件`tms_evaluation.py`中相应的内容
  - 示例：
  ```python
  # func(x) = 3.0/resultA_x + 1.0 / resultB_B2/B1_x
  func = lambda all_csv, area_index : 3.0 / all_csv['resultA_' + str(area_index)] + 1.0 / all_csv['resultB_B2/B1_' + str(area_index)]
  ```
- 修改 dataPath
  - 修改文件数据文件所在路径

#### 3.2.2 开始计算
可能需要较长时间用于统计分析
```bash
cd ./TMS_OPA
python ./tms_opa.py -o {path_to_save_result}
```

## 4 License

If you use TMS_OPA in an academic work, please cite:
  
    @article{TMS_OPA,
      title={xxxx},
      author={xxxx},
      journal={xxxx}, 
      volume={xx},
      number={xx},
      pages={xx-xx},
      year={2022}
     }