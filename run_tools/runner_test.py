import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from run_utils.builder import ConfigBuilder
from utils.tools import to_device
from time import perf_counter
from run_utils.logger import get_root_logger, print_log
import time
from relat_depth_models.load_reldepth_model import load_reldepth_model
from utils.analysis import evaluate_model_statistically, evaluate_transparent_region_ratio
import pandas as pd



def test(args):

    # config
    cfg_file = args.cfg
    # logger.info(f'Building cfg from {cfg_file}')
    with open(cfg_file, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    # builder
    builder = ConfigBuilder(args, **cfg_params)
    # logger
    states_dir = builder.get_states_dir()
    log_file = os.path.join(states_dir, f'{args.mode}_{args.logname}.log')
    logger = get_root_logger(log_file=log_file, name=args.mode)
    builder.get_logger_name(args.mode)
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    logger.info(f"====================={timestamp}=======================")
    print_log(f'Building cfg from {cfg_file}', logger=logger)
    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model
    model = builder.get_model()
    model.to(device)
    # dataloader
    test_datalader = builder.get_dataloader(split='test')
    # checking checkpoints
    checkpoint_file = args.checkpoints #os.path.join(states_dir, 'checkpoint4.tar') #results
    logger.info(f'Using checkpoint: {checkpoint_file}')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Checkpoint {checkpoint_file} (epoch {start_epoch}) loaded.")
    else:
        raise FileNotFoundError('No checkpoint')
    # metrics
    metrics = builder.get_metrics()
    # trainer
    trainer = builder.get_trainer()

    if args.reldepth_model is not None:
        reldepth_model = load_reldepth_model(args.reldepth_model, device)
    
    logger.info('start testing')
    model.eval()
    metrics.clear()
    running_time = []

    all_refract_errors = []
    all_reflect_errors = []
    all_normal_errors = []

    region_ratios = []
    region_counts = []

    with tqdm(test_datalader) as pbar:
        for data_dict in pbar:
            data_dict = to_device(data_dict, device)
            with torch.no_grad():
                if args.reldepth_model is not None:
                    time_start = perf_counter()
                    if args.reldepth_model == 'depthanything':
                        relative_depth = reldepth_model.forward(data_dict['rgb_relat']).unsqueeze(1)
                    elif args.reldepth_model == 'leres':
                        relative_depth = reldepth_model.inference(data_dict['rgb_relat'])
                    trainer(model, data_dict, relative_depth)
                    time_end = perf_counter()
                else:
                    time_start = perf_counter()
                    # print(torch.min(data_dict['depth']))
                    trainer(model, data_dict)
                    time_end = perf_counter()
                
                # 误差统计
                # batch_refract, batch_reflect, batch_normal = evaluate_model_statistically(data_dict)
                # all_refract_errors.extend(batch_refract)
                # all_reflect_errors.extend(batch_reflect)
                # all_normal_errors.extend(batch_normal)

                #占比统计
                # (ratio, count) = evaluate_transparent_region_ratio(data_dict)
                # region_ratios.append(ratio)
                # region_counts.append(count)
                
                
                _ = metrics.evaluate_batch(data_dict, record=True)

            duration = time_end - time_start
            pbar.set_description('Time: {:.4f}s'.format(duration))
            running_time.append(duration)
    avg_running_time = np.stack(running_time).mean()
    logger.info('Finish testing process, average running time: {:.4f}s'.format(avg_running_time))
    metrics_result = metrics.get_results()
    metrics.display_recorder_results()

    ##### 预测误差分析
    # 汇总为一维数组
    # refract_all = np.concatenate(all_refract_errors)
    # reflect_all = np.concatenate(all_reflect_errors)
    # normal_all = np.concatenate(all_normal_errors)

    # def compute_stats(arr):
    #     return {
    #         'Count': len(arr),
    #         'MAE': np.mean(arr),
    #         'RMSE': np.sqrt(np.mean(arr**2)),
    #         'Std': np.std(arr),
    #         'Max': np.max(arr)
    #     }

    # stats = {
    #     'Refracted': compute_stats(refract_all),
    #     'Reflected': compute_stats(reflect_all),
    #     'Normal': compute_stats(normal_all)
    # }

    # df = pd.DataFrame(stats).T
    # print("\n全数据集误差统计结果（按透明区域分类）：")
    # print(df.to_string(float_format="%.4f"))

    # # 可选：保存 CSV
    # df.to_csv("transparent_error_statistics.csv")

    ##### 占比统计
    # region_ratios = np.array(region_ratios)
    # region_counts = np.array(region_counts)

    # mean_ratios = np.mean(region_ratios, axis=0)
    # total_counts = np.sum(region_counts, axis=0)

    # print("\n=== 全数据集透明区域占比统计 ===")
    # print(f"折射区域: {mean_ratios[0]*100:.2f}% ({total_counts[0]} px)")
    # print(f"反射区域: {mean_ratios[1]*100:.2f}% ({total_counts[1]} px)")
    # print(f"正常区域: {mean_ratios[2]*100:.2f}% ({total_counts[2]} px)")
    return metrics_result

