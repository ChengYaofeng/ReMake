set -x
set -e

# remember to change the device when launch sh file
# if multi process ddp, remember to change the Port to be different
export CUDA_VISIBLE_DEVICES=0 #remember_to_change
export MASTER_ADDR=localhost
export MASTER_PORT=12355

#REMEMBER: check the config file, logname and expname
python ./main.py -m 'ddp_train' \
                -c 'configs/ddp_train/transcg_val_transcg_remake.yaml' \
                --reldepth_model 'depthanything' \
                --logname 'abaltion_no_mask_depth' \
                --expname 'abaltion_no_mask_depth'

# python ./main.py -m 'train' -c 'configs/train_transcg_val_transcg.yaml'
#configs/train_cgsyn_val_cgsyn_dfuse.yaml
#configs/train_transcg_val_transcg_dfuse.yaml
