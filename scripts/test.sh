set -x
set -e

export CUDA_VISIBLE_DEVICES=0

#REMEMBER: check the config file, logname and expname
python ./main.py -m 'test' \
                --logname 'test' \
                --expname 'split_test' \
                --reldepth_model 'depthanything' \
                -c 'configs/test/transcg_remake.yaml' \
                --checkpoints 'results/00_useful_results/624_split_rel/checkpoint39.tar'


