# Paper Implementation

##  1. Intro
This repo aims to reproduce the results in reference to the [official repo](https://github.com/Owen-Liuyuxuan/visualDet3D) for the purpose of studying Ground-aware Monocular 3D Object Detection for Autonomous Driving

We used Kitti DataSet to achieve results consistent with this paper and mimic the architecture of the open source in addition to modifying the config file to be the same as the paper's training environment.
We implemented a paper in Google Colab and solved the problem by changing several codes and files to address errors caused by different environments and conditions. 

For example, when implementing Depth prediction, there was a problem that the training dataset exceeded the capacity of the Colab allowed disk. 
Therefore, we solved it by training using only some of the existing datasets. 

With this project, we were able to quickly customize GitHub's open source and develop the ability to use it.


## 2. Related Paper

This repo contains the implementation of 2021 *RAL* \& *ICRA* paper [**Ground-aware Monocular 3D Object Detection for Autonomous Driving**](https://ieeexplore.ieee.org/document/9327478). [Arxiv Page](https://arxiv.org/abs/2102.00690). 
```
@ARTICLE{9327478,
  author={Y. {Liu} and Y. {Yuan} and M. {Liu}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Ground-aware Monocular 3D Object Detection for Autonomous Driving}, 
  year={2021},
  doi={10.1109/LRA.2021.3052442}}
```
## 3. Key Features

- [Mono3D](demos/mono3D/)
- [Mono3D Video](demos/mono3D_video/)
- [Depth Predictions](demos/mono_Depth/)



## 4. Setup

To install requirements:
```setup
pip install -r requirement.txt
```
To check dependencies:
```setup
# build ops (deform convs and iou3d), We will not install operations into the system environment
./make.sh
```

## 5. Training

Please check the corresponding task: 
- Mono3D[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/mono3D/Final_Visualize_test_3D_MONO.ipynb)
- Mono3D video[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/mono3D_video/Final_Visualize_test_3D_video_MONO.ipynb)
- Depth Prediction[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/mono_Depth/Final_Visualize_test_3d_MONO_depth.ipynb)

### 5. 1 Mono3D 
#### • Data

Download the list below:
|Name|Description|
|--|--|
|image_2|Download left color images of object data set (12 GB)|
|image_3|Download right color images, if you want to use stereo information (12 GB)|
|calib|Download camera calibration matrices of object data set (16 MB)|
|labels_2| Download training labels of object data set (5 MB)|


#### • Config and Path setup

Please modify the path and other parameters in **config/\*.py**. **config/\*_example** files are templates.

**Notice**:
*_examples are **NOT** utilized by the code and \*.py under /config is **ignored** by .gitignore.

The content of the selected config file will be recorded in tensorboard at the beginning of training.

**important paths to modify in config** :
1. cfg.path.data_path: Path to KITTI training data. We expect calib, image_2, image_3, label_2 being the subfolder (directly unzipping the downloaded zips will be fine)
2. cfg.path.test_path: Path to KITTI testing data.  We expect calib, image_2 being the subfolder.
3. cfg.path.visualDet3D_path: Path to the "visualDet3D" directorty of the current repo
4. cfg.path.project_path: Path to the workdirs of the projects (will have temp_outputs, log, checkpoints)

### 5. 2 Depth Prediction
#### • Data

Download the list below:
|Name|Description|
|--|--|
|raw_data|[raw data downloader](https://github.com/Deepak3994/Kitti-Dataset.git)|
|data_depth_annotated|Download annotated depth maps data set (14 GB)|
|data_depth_selection|Download manually selected validation and test data sets (5 GB)|


#### • Config and Path setup

Please modify the path and other parameters in **config/\*.py**. **config/\*_example** files are templates.

1. cfg.path.raw_path : Path to KITTI raw data.
2. cfg.path.depth_path : Path to KITTI data depth annotated. We expect train, val being subfolder.
3. path.validation_path : Path to KITTI val_selection_cropped.
4. cfg.path.test_path : Path to KITTI test_depth_prediction_anonymous.

Please check the template's comments and other comments in codes to fully exploit the repo.


## 6. Further Info and Bug Issues

1. Open issues on the repo if you meet troubles or find a bug or have some suggestions.
2. Email to jaewoo4710@gmail.com, rmsgn100@gmail.com, morethanparadisee@gmail.com

## 7. Evaluation

Please check the corresponding task: 
- [Mono3D](demos/mono3D/)
- [Mono3D Video](demos/mono3D_video/)
- [Depth Predictions](demos/mono_Depth/)


## 8. Pre-trained Models

You can download pretrained models here:

- [Our awesome model](https://github.com/mnshtxp/Proj2_visualDet3D/releases/tag/1.0)


## 9. Results

<img src = "https://github.com/mnshtxp/Proj2_visualDet3D/blob/main/docs/result.gif?raw=true">

## 10. Related Codes
- [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN)
