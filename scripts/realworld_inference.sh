set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python ./main.py -m 'realworld' \
                --reldepth_model 'depthanything' \
                -c 'configs/inference/remake.yaml' \
                --checkpoints 'results/00_useful_results/706_cleargrasp_l1/checkpoint29.tar'
