import os
import yaml
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from run_utils.logger import get_root_logger
from run_utils.builder import ConfigBuilder
from utils.tools import to_device, display_results
from time import perf_counter
from relat_depth_models.load_reldepth_model import load_reldepth_model
import torch.nn.functional as F
from utils.visualization import plot_realat_depth
from utils.data_preparation import plot_rgb_image
LOSS_INF = 1e18

def train(args):
    
    # config
    cfg_file = args.cfg
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
    
    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model
    logger.info('Building models ...')
    model = builder.get_model()
    model.to(device)
    
    # dataloader
    train_dataloader = builder.get_dataloader(split='train', data_type='synthetic-test')
    test_dataloader = builder.get_dataloader(split='test', data_type='real-test')
    # checking checkpoints
    start_epoch = 0
    max_epoch = builder.get_max_epoch()
    checkpoint_file = os.path.join(states_dir, 'checkpoint.tar') #stats
    
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        checkpoint_metrics = checkpoint['metrics']
        checkpoint_loss = checkpoint['loss']
        logger.info(f"Checkpoint {checkpoint_file} (epoch {start_epoch}) loaded.")
    
    # load reldepth model
    if args.reldepth_model is not None:
        reldepth_model = load_reldepth_model(args.reldepth_model, device)
    
    # optimizeer schedulers
    resume = (start_epoch > 0)
    optimizer = builder.get_optimizer(model, resume=resume, resume_lr=builder.get_resume_lr())
    lr_scheduler = builder.get_lr_scheduler(optimizer, resume=resume, resume_epoch=(start_epoch - 1) if resume else None)
    if lr_scheduler is not None:
        logger.info(f'Leraning rate: {lr_scheduler.get_last_lr()}')
        
    if start_epoch != 0:
        min_loss = checkpoint_loss
        min_loss_epoch = start_epoch
        display_results(checkpoint_metrics, logger)
    else:
        min_loss = LOSS_INF
        min_loss_epoch = None

    # metrics
    metrics = builder.get_metrics()
    criterion = builder.get_criterion()
    # trainer
    trainer = builder.get_trainer()
    
     # training
    for epoch in range(start_epoch, max_epoch):
        logger.info(f'----> Epoch {epoch + 1}/{max_epoch}')
        # train
        logger.info(f'Start training process in epoch {epoch + 1}.')
        model.train()
        losses = []
        log_idx = 0
        # with torch.autograd.profiler.profile(use_cuda=True) as prof: #测试效率
            
        with tqdm(train_dataloader) as pbar:
            for data_dict in pbar:
                time_start = perf_counter()
                optimizer.zero_grad()
                data_dict = to_device(data_dict, device)
                
                if args.reldepth_model is not None:
                    with torch.no_grad():
                        if args.reldepth_model == 'depthanything':
                            relative_depth = reldepth_model.forward(data_dict['rgb_relat']).unsqueeze(1)
                        elif args.reldepth_model == 'leres':
                            relative_depth = reldepth_model.inference(data_dict['rgb_relat'])
                    trainer(model, data_dict, relative_depth)
                
                else:
                    trainer(model, data_dict)
                loss_dict = criterion(data_dict)
                
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
                
                if 'smooth' in loss_dict.keys():
                    pbar.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.8f}, smooth loss: {loss_dict['smooth'].item():.8f}")
                    
                else:
                    pbar.set_description(f'Epoch {epoch + 1}, loss: {loss.item():.8f}')
                losses.append(loss.mean().item())
                # step loss print
                if log_idx % 100 == 0:
                    logger.info(f'step {log_idx} current loss is {loss.mean():.8f}')
                log_idx += 1

            mean_loss = np.stack(losses).mean()
            logger.info(f'Finish training process in epoch {epoch + 1}, mean training loss: {mean_loss:.8f}')
            
        # test
        logger.info(f'Start testing process in epoch {epoch + 1}.')
        model.eval()
        metrics.clear()
        running_time = []
        losses = []
        with tqdm(test_dataloader) as pbar:
            for data_dict in pbar:
                data_dict = to_device(data_dict, device)
                with torch.no_grad():
                    if args.reldepth_model is not None:
                        if args.reldepth_model == 'depthanything':
                            relative_depth = reldepth_model.forward(data_dict['rgb_relat']).unsqueeze(1)
                        elif args.reldepth_model == 'leres':
                            relative_depth = reldepth_model.inference(data_dict['rgb_relat'])
                        trainer(model, data_dict, relative_depth)
                    
                    else:
                        trainer(model, data_dict)
                        
                    time_start = perf_counter()
                    loss_dict = criterion(data_dict)
                    time_end = perf_counter()
                
                    loss = loss_dict['loss']
                    _ = metrics.evaluate_batch(data_dict, record=True)
                duration = time_end - time_start
                if 'smooth' in loss_dict.keys():
                    pbar.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.8f}, smooth loss: {loss_dict['smooth'].item():.8f}")
                else:
                    pbar.set_description(f'Epoch {epoch + 1}, loss: {loss.item():.8f}')
                losses.append(loss.item())
                running_time.append(duration)
        mean_loss = np.stack(losses).mean()
        avg_running_time = np.stack(running_time).mean()
        logger.info(f'Finish testing process in epoch {epoch + 1}, mean testing loss: {mean_loss:.8f}, average running time: {avg_running_time:.4f}s')
        metrics_result = metrics.get_results()
        metrics.display_recorder_results()
                
        # update step
        if lr_scheduler is not None:
            lr_scheduler.step()
        criterion.step()
        # save model
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if builder._multigpu() else model.state_dict(),
            'loss': mean_loss,
            'metrics': metrics_result
        }
        torch.save(save_dict, os.path.join(states_dir, f'checkpoint.tar'))
        if mean_loss < min_loss:
            min_loss = mean_loss
            min_loss_epoch = epoch + 1
            torch.save(save_dict, os.path.join(states_dir, f'checkpoint{epoch + 1}.tar'))
            logger.info(f'save current model as the best model...')
            logger.info(f"save path: {os.path.join(states_dir, f'checkpoint{epoch + 1}.tar')}")
    logger.info(f'Training Finished. Min testing loss: {min_loss:.6f}, in epoch {min_loss_epoch}')