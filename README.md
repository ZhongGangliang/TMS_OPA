# TMS_OPA: A high-quality ‘standard’ average stimulation effects-based optimal coil placement

## 0 Introduction

We provided an  OPA-building pipeline for the optimal coil pose and the data from the simulation results are openly available in [Science Data Bank](https://www.doi.org/10.57760/sciencedb.02400)

- Function1: Coil Pose Conversion Tool
- Function2: Coil Pose mapping tool

## 1 Installation and Usage

1. Install and create a python environment, such as conda
2. To install the dependency library, see requirements.txt

## 2 Coil Pose Conversion Tool
### 2.1 Function
- Convert HAC coil coordinate system to MNI coordinate system inferior pose


### 2.2 Usage
```bash
cd ./TMS_OPA
python ./tms_convert.py -p {a} {b}
```
### 2.3 Results output
- MNI coordinates of the position of the center point of the coil

## 3 Coil pose mapping tool
### 3.1 Function
- Statistical analysis of simulation results to generate group coil pose mapping
### 3.2 Usage
#### 3.2.1 Modify the contents of analysis_config.py
- The subject of the simulation results and the directory where the results are stored
  - Modify the corresponding content in the file `analysis_config.py`
  - Example：
  ```python
  subjectnum_result = [
        {
            'subject': '003', # The first subject
            'simu_result_path': 'simu_result20201226232724_21443' # Corresponding simulation results catalog
        },
        {
            'subject': '004',
            'simu_result_path': 'simu_result20210124015948_33635'
        }
    ]
  ```
- Evaluation index parameters
  - Modify the contents of the file `analysis_config.py` accordingly, see the comments for details
- Objective function
  - Modify the corresponding content in the file `tms_evaluation.py`
  - Example：
  ```python
  # func(x) = 3.0/resultA_x + 1.0 / resultB_B2/B1_x
  func = lambda all_csv, area_index : 3.0 / all_csv['resultA_' + str(area_index)] + 1.0 / all_csv['resultB_B2/B1_' + str(area_index)]
  ```
- Modify dataPath
  - Modify the path where the file data file is located

#### 3.2.2 Start calculation
May take longer for statistical analysis
```bash
cd ./TMS_OPA
python ./tms_opa.py -o {path_to_save_result}
```

## 4 License

If you use TMS_OPA in an academic work, please cite:
  
    @article{TMS_OPA,
      title={Stimulation Effects Mapping for Optimal Coil Pose Atlas of Transcranial Magnetic Stimulation},
      author={Gangliang Zhong, Liang Ma, Yongfeng Yang, Baogui Zhang, Xuefeng Lu, Dan Cao, Jin Li, Nianming Zuo, Lingzhong Fan, Zhengyi Yang, Tianzi Jiang},
      journal={under review}, 
      volume={under review},
      number={under review},
      pages={under review},
      year={2022}
     }