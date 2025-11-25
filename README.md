# Rethinking Transparent Object Grasping: Depth Completion with Monocular Depth Estimation and Instance Mask

## Official Web & Paper
https://chengyaofeng.github.io/ReMake.github.io/

## Env

```
conda env create -f env.yaml

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# possible ERROR
ModuleNotFoundError: No module named 'Imath’

imath cannot install: `pip install OpenEXR`
```

## Dataset

```
dataset access:
cleargrasp: https://sites.google.com/view/cleargrasp/data
transcg: https://graspnet.net/transcg
OOD: https://drive.google.com/drive/folders/1wCB1vZ1F3up5FY5qPjhcfSfgXpAtn31H?usp=sharing
#OOD currently not avalible in official
```

```
input transcg into the datasets file as:

datasets
├── transcg
│   └── transcg
│       ├── camera_intrinsics
│       ├── metadata.json
│       ├── models
│       ├── scene1
│       ├── scene10
│       ├── scene100
│       ├── scene101
    ...
│       └── T_camera2_camera1.npy
└── transcg.py (existing file)

```

## Train
Time: 3090x1 80hours
```
cd remake

bash ./scripts/train.sh
# set the correct config file in configs/train/xxxx.yaml
# set log file name and exp name for saving results
```

## DDP Train (Recommended)
Time: 3090x8 10hours
```
cd remake

bash ./scripts/ddp_train.sh
# for multi-gpu users
# chose correct config file for target dataset and model
# set log file name and exp name for saving results
```

## Eval & Inference

```
cd remake

# eval
bash ./scripts/test.sh

# inference
bash ./scripts/inference.sh

# real-world inference
bash ./scripts/realworld_inference.sh
# for realsense-d435 users
```

## Cite
```
@article{cheng2025rethinking,
  title={Rethinking Transparent Object Grasping: Depth Completion with Monocular Depth Estimation and Instance Mask},
  author={Cheng, Yaofeng and Gao, Xinkai and Zhang, Sen and Zeng, Chao and Zha, Fusheng and Sun, Lining and Yang, Chenguang},
  journal={arXiv preprint arXiv:2508.02507},
  year={2025}
}
```

## Thanks
Paper recommended： TDCNet official code: https://github.com/XianghuiFan/TDCNet
