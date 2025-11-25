 
import os
from typing import Callable

class ConfigBuilder():
    def __init__(self, args, **params):
        super(ConfigBuilder, self).__init__()
        
        self.params = params
        self.model_params = params.get('model', {})
        self.optimizer_params = params.get('optimizer', {})
        self.lr_scheduler_params = params.get('lr_scheduler', {})
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})
        self.trainer_params = params.get('trainer', {})
        self.metrics_params = params.get('metrics', {})
        self.states_params = params.get('states', {})
        self.inference_params = params.get('inference', {})
        self.tb_log = params.get('tb_log', {})
        self.args = args
        self.multigpu = self._multigpu()
    
    def get_logger_name(self, logger_name):
        self.logger_name = logger_name
        
    
    def get_model(self, model_params=None):

        from models.tdcnet import TDCNet
        from models.dfnet import DFNet
        from models.remake import ReMake


        
                
        if model_params is None:
            model_params = self.model_params
        model_type = model_params.get('type', 'tode')
        params = model_params.get('params', {})
        
        if model_type == 'tode':
            model = TDCNet.build(**params)
        elif model_type == 'dfnet':
            model = DFNet.build(**params)
        elif model_type == 'remake':
            model = ReMake.build(**params)
        else:
            raise NotImplementedError(f'Invalid model type: {model_type}.')
        return model
    
    def get_trainer(self, model_params=None) -> Callable:
        '''
            define trainer with correspond model
        '''
        from .trainer import df_trainer, tdc_trainer, remake_trainer
        
        if model_params is None:
            model_params = self.model_params
        model_type = model_params.get('type', 'tode')
        
        if model_type == 'tode':
            trainer = tdc_trainer
        elif model_type == 'dfnet':
            trainer = df_trainer
        elif model_type == 'remake':
            trainer = remake_trainer
        else:
            raise NotImplementedError(f'Invalid model type: {model_type}.')
        
        return trainer
            
    
    def get_states_dir(self, states_params=None):
        if states_params is None:
            states_params = self.states_params
        states_dir = states_params.get('states_dir', 'results')
        if self.args.expname is not None:
            states_exper = self.args.expname
        else:
            states_exper = states_params.get('states_exper', 'default')
        states_res_dir = os.path.join(states_dir, states_exper)
        if os.path.exists(states_res_dir) == False:
            os.makedirs(states_res_dir)
        return states_res_dir

    def get_inference_cuda_id(self, inference_params=None):
        '''
            return: cuda device {int}
        '''
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('cuda_id', 0)
    
    def get_inference_checkpoint_path(self, inference_params=None):
        '''
            return: checkpoint path {str}
        '''
        if inference_params is None:
            inference_params = self.inference_params
        if self.args.checkpoints is not None:
            checkpoints = self.args.checkpoints
        else:
            checkpoints = inference_params.get('checkpoint_path', os.path.join('checkpoints', 'checkpoint.tar'))
        return checkpoints

    def get_inference_image_size(self, inference_params=None):
        '''
            return: imagesize {typle}: {int, int}
        '''
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('image_size', (320,240))

    def get_inference_depth_min_max(self, inference_params=None):
        '''
            return: depth crop district {tuple}: {float, float}
        '''
        if inference_params is None:
            inference_params = self.inference_params
        depth_min = inference_params.get('depth_min', 0.3)
        depth_max = inference_params.get('depth_max', 1.5)
        return depth_min, depth_max
    
    def get_inference_dpeth_norm(self, inference_params=None):
        '''
            return: depth normalization coefficient: float
        '''
        if inference_params is None:
            inference_params = self.inference_params
        depth_norm = inference_params.get('depth_norm', 1.0)
        return depth_norm
    
    def get_dataset(self, dataset_params=None, split='train', data_type=None):
        '''
            return:
                Dataset
        '''
        from datasets.transcg import TransCG
        from datasets.cleargrasp import ClearGraspRealWorld, ClearGraspSynthetic
        
        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, {'type': 'transcg'})
        if type(dataset_params) == dict:
            dataset_type = str.lower(dataset_params.get('type', 'transcg'))
            if dataset_type == 'transcg':
                dataset = TransCG(split=split, **dataset_params)
            elif dataset_type == 'cleargrasp-real':
                dataset = ClearGraspRealWorld(split = split, **dataset_params, data_type=data_type)
            elif dataset_type == 'cleargrasp-syn':
                dataset = ClearGraspSynthetic(split = split, **dataset_params, data_type=data_type)
            else:
                raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
        elif type(dataset_params) == list:
            dataset_types = []
            dataset_list = []
            for single_dataset_params in dataset_params:
                dataset_type = str.lower(single_dataset_params.get('type', 'transcg'))
                if dataset_type in dataset_types:
                    raise AttributeError('Duplicate dataset found.')
                else:
                    dataset_types.append(dataset_type)
                if dataset_type == 'transcg':
                    dataset = TransCG(split = split, **single_dataset_params)
                elif dataset_type == 'cleargrasp-real':
                    dataset = ClearGraspRealWorld(split = split, **single_dataset_params)
                elif dataset_type == 'cleargrasp-syn':
                    dataset = ClearGraspSynthetic(split = split, **single_dataset_params)
                else:
                    raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
                dataset_list.append(dataset)
                # logger.info('Load {} dataset as {}ing set with {} samples.'.format(dataset_type, split, len(dataset)))
            # dataset = ConcatDataset(dataset_list)
        else:
            raise AttributeError('Invalid dataset format.')
        return dataset
                
    
    def get_dataloader(self, dataset_params=None, split='train', batch_size=None, dataloader_params=None, data_type=None):
        '''
            return:
                Dataloader
        '''
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        import random
        if batch_size is None:
            if split == 'train':
                batch_size = self.trainer_params.get('batch_size', 32)
            else:
                batch_size = self.trainer_params.get('test_batch_size', 1)
        if dataloader_params is None:
            dataloader_params = self.dataloader_params
        dataset = self.get_dataset(dataset_params, split, data_type)
        # print(f'multigpu: {self.multigpu}')
        if self.multigpu is True:
            sampler = DistributedSampler(dataset)
            sampler.set_epoch(random.randint(1, 100))
            loader = DataLoader(dataset, batch_size, sampler=sampler, pin_memory=True, **dataloader_params)
            return loader
        else:
            return DataLoader(dataset, batch_size, **dataloader_params)
            
    
    
    def get_metrics(self, metrics_params=None):
        '''
            Get the metrics setting from config
            Input:
                metrics_params: dict, optional, default{None}
            Output:

        '''
        if metrics_params is None:
            metrics_params = self.metrics_params
        metrics_list = metrics_params.get('types', ['MSE', 'MaskedMSE', 'RMSE', 'MaskedRMSE',
                                                    'REL', 'MaskedREL', 'MAE', 'MaskedMAE',
                                                    'Threshold@1.05', 'MaskedThreshold@1.05',
                                                    'Threshold@1.10', 'MaskedThreshold@1.10',
                                                    'Threshold@1.25', 'MaskedThreshold@1.25'])
        from utils.metrics import MetricsRecorder
        metrics = MetricsRecorder(logger_name=self.logger_name, metrics_list=metrics_list, **metrics_params)
        return metrics
    
    
    def get_max_epoch(self, trainer_params=None):
        '''
            get cfg max epoch
        '''
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('max_epoch', 40)
    
    
    def _multigpu(self, trainer_params=None):
        '''
            multigpu training
        '''
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('multigpu', False)


    def get_criterion(self, criterion_params=None):
        '''
            get cfg criterion
        '''
        if criterion_params is None:
            criterion_params = self.trainer_params.get('criterion', {})
        loss_type = criterion_params.get('type', 'custom_masked_mse_loss')
        from utils.criterion import Criterion
        criterion = Criterion(**criterion_params)
        return criterion
    
    
    def get_optimizer(self, model, optimizer_params=None, resume=False, resume_lr=None):
        '''
            get cfg optimizer
        '''
        from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
        if optimizer_params is None:
            optimizer_params = self.optimizer_params
        type = optimizer_params.get('type', 'AdamW')
        params = optimizer_params.get('params', {})
        if resume:
            network_params = [{'params': model.parameters(), 'initial_lr': resume_lr}]
            params.update(lr = resume_lr)
        else:
            network_params = model.parameters()
        if type == 'SGD':
            optimizer = SGD(network_params, **params)
        elif type == 'ASGD':
            optimizer = ASGD(network_params, **params)
        elif type == 'Adagrad':
            optimizer = Adagrad(network_params, **params)
        elif type == 'Adamax':
            optimizer = Adamax(network_params, **params)
        elif type == 'Adadelta':
            optimizer = Adadelta(network_params, **params)
        elif type == 'Adam':
            optimizer = Adam(network_params, **params)
        elif type == 'AdamW':
            optimizer = AdamW(network_params, **params)
        elif type == 'RMSprop':
            optimizer = RMSprop(network_params, **params)
        else:
            raise NotImplementedError('Invalid optimizer type.')
        return optimizer
    
    
    def get_lr_scheduler(self, optimizer, lr_scheduler_params=None, resume=False, resume_epoch=None):
        '''
            get lr scheduler from cfg
        '''
        from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, LambdaLR, StepLR
        if lr_scheduler_params is None:
            lr_scheduler_params = self.lr_scheduler_params
        type = lr_scheduler_params.get('type', '')
        params = lr_scheduler_params.get('params', {})
        if resume:
            params.update(last_epoch = resume_epoch)
        if type == 'MultiStepLR':
            scheduler = MultiStepLR(optimizer, **params)
        elif type == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, **params)
        elif type == 'CyclicLR':
            scheduler = CyclicLR(optimizer, **params)
        elif type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, **params)
        elif type == 'LambdaLR':
            scheduler = LambdaLR(optimizer, **params)
        elif type == 'StepLR':
            scheduler = StepLR(optimizer, **params)
        elif type == '':
            scheduler = None
        else:
            raise NotImplementedError('Invalid learning rate scheduler type.')
        return scheduler
    
    
    def get_resume_lr(self, trainer_params=None):
        '''
        get resume lrarning rate from cfg
        '''
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('resume_lr', 0.001)
    
    