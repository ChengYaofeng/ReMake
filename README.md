# Yiedling Fusion Transparent Object Depth Completion for Robot Grasping

# Env
```

# 如果没有创建用这个
conda env create -f env.yaml
# 创建之后用这个，后续的包运行时缺啥下载啥
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# ERROR
imath安装不上的原因可能和`pip install OpenEXR`
ModuleNotFoundError: No module named 'Imath’
```

# Dataset
```
# put all the transcg dataset as follows in datasets file
├── transcg (dataset)
│   └── transcg
│       ├── camera_intrinsics
│       ├── metadata.json
│       ├── models
│       ├── scene1
│       ├── scene10
│       ├── scene100
│       ├── scene101
    ...
│       └── T_camera2_camera1.npy
└── transcg.py (existing file)
```

# Train
```
cd YFTrans

记得保留数据集
```

# Add to config
```
directly use the file in config_change
```
<details>
<summary>点击展开查看日志</summary>
> Log
> 
    # Mar 31
    model createing: 

    # April 1
    test & dataset

    # April 2
    test finished
    Issues: Metric缺失， train缺失， states缺失，明天研究如何训练

    # April 3
    test finish
    Issues: not log file & start train ref: transcg

    # April 4
    train
    Issues: bug with train in criterion

    # April 6
    train finish
    Issues: bug with numworkers & model refine

    # April 7
    all model finish

    # April 10
    new_train_fuse

    # April 11
    cg dataloader;
    fuse work but still have problem
    Question: cg dataset can't resize

    Finding: delete the wrong points in transcg may help a lot

    # April 14
    realworld, expname, cgrasp
    TODO: cgrasp dataset, moculardepth refine, multigpu

    # April 16
    cgrasp can run, cuda device, multigpu, transcg model
    TODO: mocular depth method change, fuse method design, decorator

    # April 17
    decorator, bug_fix: ddp train based previous model
    TODO: mocular depth method change, fuse method design

    # April 22
    new model: mlp-concat&plus feature

    # April 23
    new model: transformer

    # April 24
    new model: resnet
    chamferdist

    Best model: transformer plus

    # April 25
    new model: cross attention # no rgb information, only relat & depth information corss

    # April 28
    TODO: 1. annotate chamferdist, 2. Crossattn 3. cleargrasp

    # April 29
    crossattn, cleargrasp

    # April 30
    window cross is bad, global cross attn seems ok, chamferdist is time consuming
    
    # May 1
    Realworld depth estimation's error will lead the object incline


    # May 9
    Dataset check, 435 better than 455, background should not contain too far or too near, data aug
    # TODO: data aug training

    # May 12
    Correct the relate depth generate: preprocess
    # TODO: test model and realworld exp

    # May 14
    MFFM test, Ori decoder train
    # TODO cross attention but no large patch

    # May 15
    DEEP decoder test, L1 a little worse, mask better than MFFM

    # May 16
    mask_concat to depth

    # May 22
    cleargrasp train

    # May 23
    cleargrasp keep depth; res model

    # May 27
    add sem segmentation & leres reldepth pred & multi rel code structure

    # May 28
    leres preprocess is not correct
    
    
</details>


