#!/bin/bash
set -x
set -e

# remember to change the device when launch sh file
export CUDA_VISIBLE_DEVICES=0

#REMEMBER: check the config file, logname and expname
python ./main.py -m 'train' \
                -c 'configs/train/cgsyn_val_cgsyn_remake.yaml' \
                --reldepth_model 'depthanything' \
                --logname 'cg' \
                --expname 'cg'

