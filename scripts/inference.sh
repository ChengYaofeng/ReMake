set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python ./main.py -m 'inference' \
                --reldepth_model 'depthanything' \
                -c 'configs/inference/remake.yaml' \
                --checkpoints ''


