# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import re
import json
import tempfile
import torch
import stylegan2ada.dnnlib as dnnlib
import wandb
import utils
import time
import numpy as np
import omegaconf
import copy
import PIL.Image
import psutil
import pickle

from stylegan2ada.torch_utils import misc
from stylegan2ada.torch_utils import training_stats
from stylegan2ada.torch_utils import custom_ops
from stylegan2ada.torch_utils.ops import conv2d_gradfix
from stylegan2ada.torch_utils.ops import grid_sample_gradfix

import stylegan2ada.legacy as legacy
from stylegan2ada.metrics import metric_main

from train_parts.generators import generators
from train_parts.discriminators import discriminators
from train_parts.datasets import datasets
from train_parts.augmentations import augmentations
from train_parts.losses import losses
from train_parts.losses_base import losses_arch
from train_parts.optimizers import optimizers
from train_parts_args.augmentaions_agrs import augpipe_specs
from dataclasses import asdict
from collections import defaultdict


#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------
import os
import torch
import json
from omegaconf import OmegaConf

trainers = utils.ClassRegistry()

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

        
def printer(obj, tabs=0):
    for (key, value) in obj.items():
        try:
            _ = value.items()
            print(' ' * tabs + str(key) + ':')
            printer(value, tabs + 4)
        except:
            print(f' ' * tabs + str(key) + ' : ' + str(value))
            
def to_easy_dict(obj):
    try:
        items = obj.items()
    except:
        if type(obj) == omegaconf.listconfig.ListConfig:
            list_of_items = []
            for item in obj:
                list_of_items.append(to_easy_dict(item))
            return list_of_items
        else:
            return obj
        
    easy_dict = dnnlib.EasyDict()
    for (key, value) in items:
        easy_dict[key] = to_easy_dict(value)
    return easy_dict
    


@trainers.add_to_registry("base")
class BaseTrainer:
    def __init__(self, config):
        self.config = config
    
    
    def setup_arguments(self):
        config = self.config
        
        if config.trans.resume == 'from_data':
            with open(os.path.join(config.trans.resume_dir, config.trans.args_name)) as json_data:
                args = dnnlib.EasyDict(json.load(json_data))
            args.resume_params = dnnlib.EasyDict(config.trans)
            self.desc = args.desc
            self.args = args
            return
        
        args = dnnlib.EasyDict()
        
        args.start_options = dnnlib.EasyDict({'cur_nimg' : 0,
        'cur_tick' : 0,
        'batch_idx' : 0,
         'wandb_step' : 0})

        # ------------------------------------------
        # General options: gpus, snap, metrics, seed
        # ------------------------------------------

        if not (config.perf.gpus >= 1 and config.perf.gpus & (config.perf.gpus - 1) == 0):
            raise UserError('--gpus must be a power of two')
        args.num_gpus = config.perf.gpus


        if config.log.snap < 1:
            raise UserError('--snap must be at least 1')
        args.image_snapshot_ticks = config.log.snap
        args.network_snapshot_ticks = config.log.snap

        if not all(metric_main.is_valid_metric(metric) for metric in config.log.metrics):
            raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
        args.metrics = list(config.log.metrics)

        args.random_seed = config.gen.seed
        
        # -----------------------------------
        # Names
        # -----------------------------------
        
        args.names = dnnlib.EasyDict()
        args.names.dataset = self.config.data.dataset
        args.names.dataloader = self.config.data.dataloader
        args.names.generator = self.config.gen.generator
        args.names.discriminator = self.config.gen.discriminator
        args.names.optim_gen = self.config.gen.optim_gen
        args.names.optim_disc = self.config.gen.optim_disc
        args.names.gen_regs = self.config.gen.gen_regs
        args.names.disc_regs = self.config.gen.disc_regs
        args.names.loss_arch = self.config.gen.loss_arch
        args.names.loss = self.config.gen.loss
        args.names.aug_type = self.config.aug.aug_type
        args.use_wandb = self.config.log.wandb
        self.gen_regs_all = self.config.gen_regs_all
        self.disc_regs_all = self.config.disc_regs_all
        args.n_dis = self.config.gen.n_dis
        
        
        
        
        
        # -----------------------------------
        # Dataset: data, cond, subset, mirror
        # -----------------------------------

        assert isinstance(config.data.dataset_path, str)
        args.dataset = config.data.dataset
        args.training_set_kwargs = dnnlib.EasyDict(path=config.data.dataset_path, **config.datasets_args[config.data.dataset])
        args.data_loader_kwargs = dnnlib.EasyDict(**config.dataloaders_args[config.data.dataloader])
        try:
            training_set = datasets[config.data.dataset](**args.training_set_kwargs)
            args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
            args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
            args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
            desc = training_set.name
            del training_set # conserve memory
        except IOError as err:
            raise UserError(f'--data: {err}')

        if config.data.cond:
            if not args.training_set_kwargs.use_labels:
                raise UserError('--cond=True requires labels specified in dataset.json')
            desc += '-cond'
        else:
            args.training_set_kwargs.use_labels = False

        if config.data.subset > 0:
            if not 1 <= config.data.subset <= args.training_set_kwargs.max_size:
                raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
            desc += f'-subset{config.data.subset}'
            if config.data.subset < args.training_set_kwargs.max_size:
                args.training_set_kwargs.max_size = config.data.subset
                args.training_set_kwargs.random_seed = args.random_seed

        if config.data.mirror:
            desc += '-mirror'
            args.training_set_kwargs.xflip = True

        # ------------------------------------
        # Base config: cfg, gamma, kimg, batch
        # ------------------------------------


        args.G_kwargs = dnnlib.EasyDict(**config.gens_args[config.gen.generator])
        args.D_kwargs = dnnlib.EasyDict(**config.discs_args[config.gen.discriminator])

        args.G_opt_kwargs = dnnlib.EasyDict(**config.optim_gen_args[config.gen.optim_gen])
        args.D_opt_kwargs = dnnlib.EasyDict(**config.optim_disc_args[config.gen.optim_disc])
        args.loss_kwargs = dnnlib.EasyDict(**config.losses_arch_args[config.gen.loss_arch])

        args.total_kimg = config.gen.kimg  # проверить батч и гпу батч на делимость
        args.batch_size = config.gen.batch  # spec.mb
        args.batch_gpu = config.gen.batch_gpu  # spec.mb // spec.ref_gpus
        args.ema_kimg = config.ema.kimg  # spec.ema
        args.ema_rampup = config.ema.ramp if config.ema.ramp != -1 else None # spec.ramp
        args.D_reg_interval = config.gen.d_reg_interval
        args.G_reg_interval = config.gen.g_reg_interval
        args.progress_fn = None
        args.abort_fn = None
        args.kimg_per_tick = self.config.log.kimg_per_tick
        args.use_ema = self.config.ema.use_ema


        # ---------------------------------------------------
        # Discriminator augmentation: aug, p, target, augpipe
        # ---------------------------------------------------

        desc += f'-{config.aug.aug}'
        args.ada_interval = 4
        args.ada_target = None
        if config.aug.aug == 'ada':
            args.ada_target = 0.6

        elif config.aug.aug == 'noaug':
            pass

        elif config.aug.aug == 'fixed':
            if config.aug.p < 0:
                raise UserError(f'--aug={config.aug.aug} requires specifying --p')

        else:
            raise UserError(f'--aug={config.aug.aug} not supported')

        args.augment_p = 0
        if config.aug.p >= 0:
            if config.aug.aug != 'fixed':
                raise UserError('--p can only be specified with --aug=fixed')
            if not 0 <= config.aug.p <= 1:
                raise UserError('--p must be between 0 and 1')
            desc += f'-p{config.aug.p:g}'
            args.augment_p = config.aug.p

        if config.aug.target >= 0:
            if config.aug.aug != 'ada':
                raise UserError('--target can only be specified with --aug=ada')
            if not 0 <= config.aug.target <= 1:
                raise UserError('--target must be between 0 and 1')
            desc += f'-target{config.aug.target:g}'
            args.ada_target = config.aug.target

            desc += f'-{config.aug.augpipe}'
        
        args.augment_kwargs = None
        if config.aug.aug != 'noaug':
            args.augment_kwargs = dnnlib.EasyDict(**asdict(augpipe_specs[config.aug.augpipe]()))

        # ----------------------------------
        # Transfer learning: resume, freezed
        # ----------------------------------

        resume_specs = {
            'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
            'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
            'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
            'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
            'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
        }

        args.resume_pkl = None
        args.ada_kimg = 500
        if config.trans.resume == 'noresume':
            desc += '-noresume'
        elif config.trans.resume in resume_specs:
            desc += f'-resume{config.trans.resume}'
            args.resume_pkl = resume_specs[config.trans.resume] # predefined url
        else:
            desc += '-resumecustom'
            args.resume_pkl = config.trans.resume_url   # config.trans.resume # custom path or url

        if config.trans.resume != 'noresume':
            args.ada_kimg = 100 # make ADA react faster at the beginning
            args.ema_rampup = None # disable EMA rampup

        if config.trans.freezed >= 0:
            desc += f'-freezed{config.trans.freezed:d}'
            args.D_kwargs.block_kwargs.freeze_layers = config.trans.freezed

        # -------------------------------------------------
        # Performance options: fp32, nhwc, nobench, workers
        # -------------------------------------------------

        args.cudnn_benchmark = True
        if config.perf.nobench:
            args.cudnn_benchmark = False
        
        args.allow_tf32 = False
        if config.perf.allow_tf32:
            args.allow_tf32 = True

        args = to_easy_dict(args)
        self.run_desc = desc
        self.args = args

#----------------------------------------------------------------------------
    
    
    def setup_logs(self):
        dnnlib.util.Logger(should_flush=True)
        
        # Pick output directory.
        prev_run_dirs = []
        if os.path.isdir(self.config.log.output): # outdir
            prev_run_dirs = [x for x in os.listdir(self.config.log.output) if os.path.isdir(os.path.join(self.config.log.output, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        self.args.run_dir = os.path.join(self.config.log.output, f'{cur_run_id:05d}-{self.run_desc}')
        assert not os.path.exists(self.args.run_dir)
        
        if self.rank == 0:
            print()
            print('Training options:')
            printer(self.args)
            print()
            print(f'Output directory:   {self.args.run_dir}')
            print(f'Training data:      {self.args.training_set_kwargs.path}')
            print(f'Training duration:  {self.args.total_kimg} kimg')
            print(f'Number of GPUs:     {self.args.num_gpus}')
            print(f'Number of images:   {self.args.training_set_kwargs.max_size}')
            print(f'Image resolution:   {self.args.training_set_kwargs.resolution}')
            print(f'Conditional model:  {self.args.training_set_kwargs.use_labels}')
            print(f'Dataset x-flips:    {self.args.training_set_kwargs.xflip}')
            print()
        
        if self.config.exp.dry_run:
            return
        
        if self.rank == 0 and self.config.trans.resume != 'from_data':
            # Create output directory.
            print('Creating output directory...')
            os.makedirs(self.args.run_dir)

            with open(os.path.join(self.args.run_dir, 'training_options.json'), 'wt') as f:
                json.dump(self.args, f, indent=2)



            dnnlib.util.Logger(file_name=os.path.join(self.args.run_dir, 'log.txt'), file_mode='a', should_flush=True)
        
        if self.args.use_wandb:
            if self.config.trans.resume == 'noresume':
                wandb_id = wandb.util.generate_id()
                self.args.wandb_id = wandb_id
                self.args.wandb_project = self.config.exp.project
                wandb.init(id=wandb_id, resume='allow', project=self.config.exp.project, entity="retir", name=self.config.exp.name)
                wandb.config = self.args
            else:
                wandb.init(id=self.args.wandb_id, resume=True, project=self.args.wandb_project, entity="retir")
        
        # Initialize logs.
        if self.rank == 0:
            print('Initializing logs...')
        self.stats_collector = training_stats.Collector(regex='.*')
        self.stats_metrics = dict()
        self.stats_jsonl = None
        self.stats_tfevents = None
        self.stats_jsonl = open(os.path.join(self.args.run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            self.stats_tfevents = tensorboard.SummaryWriter(self.args.run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)


    def distribute_torch(self, temp_dir):
        if self.args.num_gpus > 1:
            init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
            if os.name == 'nt':
                init_method = 'file:///' + init_file.replace('\\', '/')
                torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=trainer.args.num_gpus)
            else:
                init_method = f'file://{init_file}'
                torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=self.rank, world_size=self.args.num_gpus)

        # Init torch_utils.
        sync_device = torch.device('cuda', self.rank) if self.args.num_gpus > 1 else None
        training_stats.init_multiprocessing(rank=self.rank, sync_device=sync_device)
        if self.rank != 0:
            custom_ops.verbosity = 'none'
        
    
    def init_params(self):
        # Initialize.
        self.wandb_step = self.args.start_options['wandb_step']
        self.start_time = time.time()
        self.device = torch.device('cuda', self.rank)
        self.metrics_time = 0
        np.random.seed(self.args.random_seed * self.args.num_gpus + self.rank)
        torch.manual_seed(self.args.random_seed * self.args.num_gpus + self.rank)
        torch.backends.cudnn.benchmark = self.args.cudnn_benchmark              # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = self.args.allow_tf32  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = self.args.allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
        conv2d_gradfix.enabled = True                                 # Improves training speed.
        grid_sample_gradfix.enabled = True                            # Avoids errors with the augmentation pipe.
        
        
        
    def setup_dataset(self):
        if self.rank == 0:
            print('Loading training set...')
            
        self.training_set = datasets[self.args.names.dataset](**self.args.training_set_kwargs)
        
        self.training_set_sampler = misc.InfiniteSampler(dataset=self.training_set, rank=self.rank, num_replicas=self.args.num_gpus, seed=self.args.random_seed)
        self.training_set_iterator = iter(torch.utils.data.DataLoader(dataset=self.training_set, sampler=self.training_set_sampler, batch_size=self.args.batch_size//self.args.num_gpus, **self.args.data_loader_kwargs))
        if self.rank == 0:
            print()
            print('Num images: ', len(self.training_set))
            print('Image shape:', self.training_set.image_shape)
            print('Label shape:', self.training_set.label_shape)
            print()
    
    
    def setup_networks(self):
        # Construct networks.
        if self.rank == 0:
            print('Constructing networks...')

        common_kwargs = dict(c_dim=self.training_set.label_dim, img_resolution=self.training_set.resolution, img_channels=self.training_set.num_channels)
        self.G = generators[self.args.names.generator](**self.args.G_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device)
        self.D = discriminators[self.args.names.discriminator](**self.args.D_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device)
        self.G_ema = None
        if self.args.use_ema:
            self.G_ema = copy.deepcopy(self.G).eval()
        
        # Resume from existing pickle.
        if (self.args.resume_pkl is not None) and (self.rank == 0):
            print(f'Resuming from "{self.args.resume_pkl}"')
            with dnnlib.util.open_url(self.args.resume_pkl) as f:
                resume_data = legacy.load_network_pkl(f)
            for name, module in [('G', self.G), ('D', self.D)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
                
        # Print network summary tables.
        try:
            if self.rank == 0:
                z = torch.empty([self.args.batch_gpu, self.G.z_dim], device=self.device)
                c = torch.empty([self.args.batch_gpu, self.G.c_dim], device=self.device)
                img = misc.print_module_summary(self.G, [z, c])
                misc.print_module_summary(self.D, [img, c])
        except:
            print('Cant print model summary')
            
            
    def setup_augmentations(self):
        # Setup augmentation.
        if self.rank == 0:
            print('Setting up augmentation...')
        self.augment_pipe = None
        self.ada_stats = None
        if (self.args.augment_kwargs is not None) and (self.args.augment_p > 0 or self.args.ada_target is not None):
            self.augment_pipe = augmentations[self.args.names.aug_type](**self.args.augment_kwargs).train().requires_grad_(False).to(self.device)
            self.augment_pipe.p.copy_(torch.as_tensor(self.args.augment_p))
            if self.args.ada_target is not None:
                self.ada_stats = training_stats.Collector(regex='Loss/signs/real')
     
    
    def distrib_acrros_gpu(self):
        if self.rank == 0:
            print(f'Distributing across {self.args.num_gpus} GPUs...')
        self.ddp_modules = dict()
        for name, module in [('G', self.G), ('D', self.D), (None, self.G_ema), ('augment_pipe', self.augment_pipe)]:
            if (self.args.num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[self.device], broadcast_buffers=False)
                module.requires_grad_(False)
            if name is not None:
                self.ddp_modules[name] = module
                
                
                
    def setup_training_phases(self):
        if self.rank == 0:
            print('Setting up training phases...')
        self.loss = losses_arch[self.args.names.loss_arch](device=self.device,
                                                       loss=self.args.names.loss,
                                                       gen_regs = [(key, self.gen_regs_all[key]) for key in self.args.names.gen_regs],
                                                       dis_regs = [(key, self.disc_regs_all[key]) for key in self.args.names.disc_regs],
                                                       **self.ddp_modules,
                                                       **self.args.loss_kwargs)

        intervals = defaultdict(lambda:1)
        intervals['G'] = self.args.n_dis         
        
        self.phases = []
        for name, module, opt_kwargs, reg_interval, opt_name in [('G', self.G, self.args.G_opt_kwargs, self.args.G_reg_interval, self.args.names.optim_gen), ('D', self.D, self.args.D_opt_kwargs, self.args.D_reg_interval, self.args.names.optim_disc)]:
            if reg_interval <= 0:
                self.opt = optimizers[opt_name](params=module.parameters(), **opt_kwargs)
                self.phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=self.opt, interval=intervals[name])]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                self.opt = optimizers[opt_name](params=module.parameters(), **opt_kwargs)
                self.phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=self.opt, interval=1)]
                self.phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=self.opt, interval=reg_interval)]
        for phase in self.phases:
            phase.start_event = None
            phase.end_event = None
            if self.rank == 0:
                phase.start_event = torch.cuda.Event(enable_timing=True)
                phase.end_event = torch.cuda.Event(enable_timing=True)
                
                
    def save_snapshot(self, cur_nimg):
        snapshot_pkl = None
        snapshot_data = None
        snapshot_data = dict(training_set_kwargs=dict(self.args.training_set_kwargs))
        for name, module in [('G', self.G), ('D', self.D), ('G_ema', self.G_ema), ('augment_pipe', self.augment_pipe)]:
            try: 
                if module is not None:
                    if self.args.num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            except:
                if self.rank == 0:
                    print(f'Cannot deepcopy module {name}, skip')
        snapshot_pkl = os.path.join(self.args.run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
        if self.rank == 0:
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
        return snapshot_data, snapshot_pkl
    
    
    def evaluate_metrics(self, snapshot_data, snapshot_pkl):
        if self.rank == 0:
            print('Evaluating metrics...')
        self.metrics_time = 0
        for metric in self.args.metrics:
            result_dict = metric_main.calc_metric(metric=metric, dataset_name=self.args.names.dataset, G=(snapshot_data['G_ema'] if self.args.use_ema else snapshot_data['G']),
                dataset_kwargs=self.args.training_set_kwargs, num_gpus=self.args.num_gpus, rank=self.rank, device=self.device)
            self.metrics_time += result_dict['total_time']
            if self.args.use_wandb:
                if metric == 'fid50k_full':
                    wandb.log({'FID': result_dict['results']['fid50k_full']}, step=self.wandb_step, commit=False)
                if metric == 'is50k':
                    wandb.log({'IS': result_dict['results']['is50k_mean']},step=self.wandb_step, commit=False)
            if self.rank == 0:
                metric_main.report_metric(result_dict, run_dir=self.args.run_dir, snapshot_pkl=snapshot_pkl)
            self.stats_metrics.update(result_dict.results)

    
    def export_sample_images(self):
        self.grid_size = None
        self.grid_z = None
        self.grid_c = None
        if self.rank == 0:
            print('Exporting sample images...')
            self.grid_size, images, labels = setup_snapshot_image_grid(training_set=self.training_set)
            save_image_grid(images, os.path.join(self.args.run_dir, 'reals.png'), drange=[0,255], grid_size=self.grid_size)
            image = PIL.Image.open(os.path.join(self.args.run_dir, 'reals.png'))
            if self.args.use_wandb:
                wandb.log({"Real images": wandb.Image(image)}, step=self.wandb_step, commit=False)
            self.grid_z = torch.randn([labels.shape[0], self.G.z_dim], device=self.device).split(self.args.batch_gpu)
            self.grid_c = torch.from_numpy(labels).to(self.device).split(self.args.batch_gpu)
            if self.args.use_ema:
                images = torch.cat([self.G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(self.grid_z, self.grid_c)]).numpy()
            else:
                self.G.eval()
                images = torch.cat([self.G(z=z, c=c, noise_mode='const').cpu() for z, c in zip(self.grid_z, self.grid_c)]).numpy()
                self.G.train()
            save_image_grid(images, os.path.join(self.args.run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=self.grid_size)
            
    
    def training_loop(self):
        if self.rank == 0:
            print(f'Training for {self.args.total_kimg} kimg...')
            print()
        cur_nimg = self.args.start_options['cur_nimg']
        cur_tick = self.args.start_options['cur_tick']
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - self.start_time
        batch_idx = self.args.start_options['batch_idx']
        if self.args.progress_fn is not None:
            self.args.progress_fn(0, self.args.total_kimg)
        while True:

            # Fetch training data.
            with torch.autograd.profiler.record_function('data_fetch'):
                phase_real_img, phase_real_c = next(self.training_set_iterator)
                phase_real_img = (phase_real_img.to(self.device).to(torch.float32) / 127.5 - 1).split(self.args.batch_gpu)
                phase_real_c = phase_real_c.to(self.device).split(self.args.batch_gpu)
                all_gen_z = torch.randn([len(self.phases) * self.args.batch_size, self.G.z_dim], device=self.device)
                all_gen_z = [phase_gen_z.split(self.args.batch_gpu) for phase_gen_z in all_gen_z.split(self.args.batch_size)]
                all_gen_c = [self.training_set.get_label(np.random.randint(len(self.training_set))) for _ in range(len(self.phases) * self.args.batch_size)]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(self.device)
                all_gen_c = [phase_gen_c.split(self.args.batch_gpu) for phase_gen_c in all_gen_c.split(self.args.batch_size)]

            # Execute training phases.
           # losses = defaultdict(list)
            for phase_idx, (phase, phase_gen_z, phase_gen_c) in enumerate(zip(self.phases, all_gen_z, all_gen_c)):
                if batch_idx % phase.interval != 0:
                    continue

                # Initialize gradient accumulation.
                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(self.device))
                phase.opt.zero_grad(set_to_none=True)
                phase.module.requires_grad_(True)

                # Accumulate gradients over multiple rounds.
                for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                    sync = (round_idx == self.args.batch_size // (self.args.batch_gpu * self.args.num_gpus) - 1)
                    gain = phase.interval
                    self.loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)
                    

                # Update weights.
                phase.module.requires_grad_(False)
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    for param in phase.module.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    phase.opt.step()
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(self.device))

            if self.G_ema is not None:
                with torch.autograd.profiler.record_function('Gema'):
                    ema_nimg = self.args.ema_kimg * 1000
                    if self.args.ema_rampup is not None:
                        ema_nimg = min(ema_nimg, cur_nimg * self.args.ema_rampup)
                    ema_beta = 0.5 ** (self.args.batch_size / max(ema_nimg, 1e-8))
                    for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                        p_ema.copy_(p.lerp(p_ema, ema_beta))
                    for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
                        b_ema.copy_(b)
                    
            # Update state.
            cur_nimg += self.args.batch_size
            batch_idx += 1

            # Execute ADA heuristic.
            if (self.ada_stats is not None) and (batch_idx % self.args.ada_interval == 0):
                self.ada_stats.update()
                adjust = np.sign(self.ada_stats['Loss/signs/real'] - self.args.ada_target) * (self.args.batch_size * self.args.ada_interval) / (self.args.ada_kimg * 1000)
                self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(misc.constant(0, device=self.device)))

            # Perform maintenance tasks once per tick.
            done = (cur_nimg >= self.args.total_kimg * 1000)
            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + self.args.kimg_per_tick * 1000):
                continue

            # Print status line, accumulating the same information in stats_collector.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
            sec_per_kimg = (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3
            sec_per_tick = tick_end_time - tick_start_time
            snapshots_left = int((self.args.total_kimg - cur_nimg / 1000) / self.args.kimg_per_tick) // self.args.image_snapshot_ticks + 1
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - self.start_time)):<12s}"]
            fields += [f"time left {dnnlib.util.format_time(int((self.args.total_kimg - cur_nimg / 1000) * sec_per_kimg) + snapshots_left * self.metrics_time):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', sec_per_tick):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', sec_per_kimg):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(self.device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            fields += [f"augment {training_stats.report0('Progress/augment', float(self.augment_pipe.p.cpu()) if self.augment_pipe is not None else 0):.3f}"]
            training_stats.report0('Timing/total_hours', (tick_end_time - self.start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - self.start_time) / (24 * 60 * 60))
            if self.rank == 0:
                print(' '.join(fields))

            # Check for abort.
            if (not done) and (self.args.abort_fn is not None) and self.args.abort_fn():
                done = True
                if self.rank == 0:
                    print()
                    print('Aborting...')

            # Save image snapshot.
            if (self.rank == 0) and (self.args.image_snapshot_ticks is not None) and (done or cur_tick % self.args.image_snapshot_ticks == 0):
                if self.G_ema:
                     images = torch.cat([self.G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(self.grid_z, self.grid_c)]).numpy()
                else:
                    self.G.eval()
                    images = torch.cat([self.G(z=z, c=c, noise_mode='const').cpu() for z, c in zip(self.grid_z, self.grid_c)]).numpy()
                    self.G.train()
                save_image_grid(images, os.path.join(self.args.run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=self.grid_size)
                image = PIL.Image.open(os.path.join(self.args.run_dir, f'fakes{cur_nimg//1000:06d}.png'))
                if self.args.use_wandb:
                    wandb.log({"Generated images": wandb.Image(image)}, step=self.wandb_step, commit=False)


            # Save network snapshot and args.
            snapshot_data = None
            snapshot_pkl = None
            if (self.args.network_snapshot_ticks is not None) and (done or cur_tick % self.args.network_snapshot_ticks == 0):
                snapshot_data, snapshot_pkl = self.save_snapshot(cur_nimg)
                self.args.start_options['cur_nimg'] = cur_nimg
                self.args.start_options['cur_tick'] = cur_tick
                self.args.start_options['batch_idx'] = batch_idx
                self.args.start_options['wandb_step'] = self.wandb_step
                if self.rank == 0:
                    with open(os.path.join(self.args.run_dir, 'training_options.json'), 'wt') as f:
                        json.dump(self.args, f, indent=2)

            # Evaluate metrics.
            if (snapshot_data is not None) and (len(self.args.metrics) > 0):
                self.evaluate_metrics(snapshot_data, snapshot_pkl)
            del snapshot_data # conserve memory

            # Collect statistics.
            for phase in self.phases:
                value = []
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
            self.stats_collector.update()
            stats_dict = self.stats_collector.as_dict()

            # Update logs.
            if self.args.use_wandb:
                wandb.log({key: value['mean'] for key, value in stats_dict.items()}, step=self.wandb_step)
            self.wandb_step += 1

            timestamp = time.time()
            if self.stats_jsonl is not None:
                fields = dict(stats_dict, timestamp=timestamp)
                self.stats_jsonl.write(json.dumps(fields) + '\n')
                self.stats_jsonl.flush()
            if self.stats_tfevents is not None:
                global_step = int(cur_nimg / 1e3)
                walltime = timestamp - self.start_time
                for name, value in stats_dict.items():
                    self.stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
                for name, value in self.stats_metrics.items():
                    self.stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                self.stats_tfevents.flush()
            if self.args.progress_fn is not None:
                self.args.progress_fn(cur_nimg // 1000, self.args.total_kimg)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
            if done:
                break
                

                
                
@trainers.add_to_registry("sg2")
class SG2Trainer(BaseTrainer):
    def distrib_acrros_gpu(self):
        if self.rank == 0:
            print(f'Distributing across {self.args.num_gpus} GPUs...')
        self.ddp_modules = dict()
        for name, module in [('G_mapping', self.G.mapping), ('G_synthesis', self.G.synthesis), ('D', self.D), ('augment_pipe', self.augment_pipe)]:
            if (self.args.num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[self.device], broadcast_buffers=False)
                module.requires_grad_(False)
            if name is not None:
                self.ddp_modules[name] = module
    

                

        