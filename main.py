import os
import torch
import torch.multiprocessing as mp
import argparse
from run_tools import train, ddp_train, test, inference, realworld_inference, live_inference


def debug():
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9501))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass


def main(args):
    
    if args.mode == 'train':
        train(args)   
    elif args.mode == 'ddp_train':     
        mp.spawn(ddp_train, args=(args,), nprocs=args.world_size)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'realworld':
        realworld_inference(args)
    elif args.mode == 'live':
        live_inference(args) 
    else:
        raise NameError("no such mode, choice: 'train','ddp_train', 'test', 'inference'.")

if __name__ == '__main__':
    
    # debug()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default=os.path.join('configs', 'train_transcg_val_transcg.yaml'), help='cfg file')
    parser.add_argument('--mode', '-m', type=str, 
                        choices=['train', 'ddp_train', 'test', 'inference', 'realworld', 'live'],
                        default=None, help='choose mode')
    # parser.add_argument('--model', type=str, choices=['tode', 'dfuse'], default=None, help='choose model')
    parser.add_argument('--logname', default=None, type=str, help='log file name')
    parser.add_argument('--expname', default=None, type=str, help='experiment file name')
    parser.add_argument('--depthsize', type=str, choices=['vits', 'vitb', 'vitl'], default='vits', help='depthanything size')
    parser.add_argument('--checkpoints', type=str, default=None, help='checkpoints path')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='multi cuda device')
    parser.add_argument('--reldepth_model', type=str, default=None,
                        choices=['depthanything', 'leres'],
                        help='rel depth model name')
    
    args = parser.parse_args()
    # print(args.world_size)
    main(args)