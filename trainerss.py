# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
# import click
import re
import json
import tempfile
import torch
import stylegan2ada.dnnlib as dnnlib
import wandb
import utils

from stylegan2ada.metrics import metric_main
from stylegan2ada.torch_utils import training_stats
from stylegan2ada.torch_utils import custom_ops

from gens_and_discs import generators, discriminators
from datasets import datasets
from augmentations import augmentations
from losses import losses
from losses_base import losses_arch
from optimizers import optimizers
from augmentaions_agrs import augpipe_specs
from dataclasses import asdict


#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------
import os
import torch
import json
from omegaconf import OmegaConf

trainers = utils.ClassRegistry()


@trainers.add_to_registry("base")
class BaseTrainer:
    def __init__(self, config):
        self.config = config
    
    
    def setup_arguments(self):
        config = self.config
        args = dnnlib.EasyDict()

        # ------------------------------------------
        # General options: gpus, snap, metrics, seed
        # ------------------------------------------

        assert isinstance(config.gen.gpus, int)
        if not (config.gen.gpus >= 1 and config.gen.gpus & (config.gen.gpus - 1) == 0):
            raise UserError('--gpus must be a power of two')
        args.num_gpus = config.gen.gpus


        assert isinstance(config.data.snap, int)
        if config.data.snap < 1:
            raise UserError('--snap must be at least 1')
        args.image_snapshot_ticks = config.data.snap
        args.network_snapshot_ticks = config.data.snap

        #assert isinstance(config.gen.metrics, list), type(config.gen.metrics)
        if not all(metric_main.is_valid_metric(metric) for metric in config.gen.metrics):
            raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
        args.metrics = list(config.gen.metrics)

        args.random_seed = config.gen.seed

        # -----------------------------------
        # Dataset: data, cond, subset, mirror
        # -----------------------------------

        assert isinstance(config.data.dataset_path, str)
        args.dataset = config.data.dataset
       # args.training_set_kwargs = dnnlib.EasyDict(path=config.data.dataset_path, use_labels=True, max_size=None, xflip=False)
        args.training_set_kwargs = dnnlib.EasyDict(path=config.data.dataset_path, **config.datasets_args[config.data.dataset])
        #args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
        args.data_loader_kwargs = dnnlib.EasyDict(**config.dataloaders_args[config.data.dataloader])
        try:
            #training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
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
            if subset < args.training_set_kwargs.max_size:
                args.training_set_kwargs.max_size = config.data.subset
                args.training_set_kwargs.random_seed = args.random_seed

        if config.data.mirror:
            desc += '-mirror'
            args.training_set_kwargs.xflip = True

        # ------------------------------------
        # Base config: cfg, gamma, kimg, batch
        # ------------------------------------

        #desc += f'-{config.exp.base_cfg}'

        cfg_specs = {
            'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
            'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
            'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
            'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
            'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
            'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        }

#         assert config.exp.base_cfg in cfg_specs
#         spec = dnnlib.EasyDict(cfg_specs[config.exp.base_cfg])
#         if config.exp.base_cfg == 'auto':
#             print('AUTO SETTINGS')
#             desc += f'{config.gen.gpus:d}'
#             spec.ref_gpus = config.gen.gpus
#             res = args.training_set_kwargs.resolution
#             spec.mb = max(min(config.gen.gpus * min(4096 // res, 32), 64), config.gen.gpus) # keep gpu memory consumption at bay
#             spec.mbstd = min(spec.mb // config.gen.gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
#             spec.fmaps = 1 if res >= 512 else 0.5
#             spec.lrate = 0.002 if res >= 1024 else 0.0025
#             spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
#             spec.ema = spec.mb * 10 / 32

        args.G_kwargs = dnnlib.EasyDict(**config.gens_args[config.gen.generator])
        args.D_kwargs = dnnlib.EasyDict(**config.discs_args[config.gen.discriminator])
        #args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
        #args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
        #args.G_kwargs.mapping_kwargs.num_layers = spec.map
        #args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
        #args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
        #args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

        #args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        args.G_opt_kwargs = dnnlib.EasyDict(**config.optim_gen_args[config.gen.optim_gen])
        #args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        args.G_opt_kwargs = dnnlib.EasyDict(**config.optim_disc_args[config.gen.optim_disc])
        #args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)
        args.loss_kwargs = dnnlib.EasyDict(**config.losses_arch_args[config.gen.loss_arch])

        args.total_kimg = config.gen.kimg  # проверить батч и гпу батч на делимость
        args.batch_size = config.gen.batch  # spec.mb
        args.batch_gpu = config.gen.batch_gpu  # spec.mb // spec.ref_gpus
        args.ema_kimg = config.ema.kimg  # spec.ema
        args.ema_rampup = config.ema.ramp if config.ema.ramp != -1 else None # spec.ramp

#         if config.exp.base_cfg == 'cifar':
#             args.loss_kwargs.pl_weight = 0 # disable path length regularization
#             args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
#             args.D_kwargs.architecture = 'orig' # disable residual skip connections

#         if config.gen.gamma >= 0:
#             desc += f'-gamma{config.gen.gamma:g}'
#             args.loss_kwargs.r1_gamma = config.gen.gamma

#         if config.gen.kimg >= 1:
#             desc += f'-kimg{config.gen.kimg:d}'
#             args.total_kimg = config.gen.kimg

#         if config.gen.batch > 0:
#             if not (config.gen.batch >= 1 and config.gen.batch % config.gen.gpus == 0):
#                 raise UserError('--batch must be at least 1 and divisible by --gpus')
#             desc += f'-batch{config.gen.batch}'
#             args.batch_size = config.gen.batch
#             args.batch_gpu = config.gen.batch // config.gen.gpus
#             args.batch_gpu = 32

        # ---------------------------------------------------
        # Discriminator augmentation: aug, p, target, augpipe
        # ---------------------------------------------------

        desc += f'-{config.aug.aug}'

        if config.aug.aug == 'ada':
            args.ada_target = 0.6

        elif config.aug.aug == 'noaug':
            pass

        elif config.aug.aug == 'fixed':
            if config.aug.p < 0:
                raise UserError(f'--aug={config.aug.aug} requires specifying --p')

        else:
            raise UserError(f'--aug={config.aug.aug} not supported')

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

    #     assert augpipe is None or isinstance(augpipe, str)
    #     if augpipe is None:
    #         augpipe = 'bgc'
    #     else:
    #         if aug == 'noaug':
    #             raise UserError('--augpipe cannot be specified with --aug=noaug')
            desc += f'-{config.aug.augpipe}'

#         augpipe_specs = {
#             'blit':   dict(xflip=1, rotate90=1, xint=1),
#             'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
#             'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
#             'filter': dict(imgfilter=1),
#             'noise':  dict(noise=1),
#             'cutout': dict(cutout=1),
#             'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
#             'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
#             'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
#             'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
#             'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
#         }

#         assert config.aug.augpipe in augpipe_specs
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

#         if config.perf.fp32:
#             args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
#             args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

#         if config.perf.nhwc:
#             args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

        if config.perf.nobench:
            args.cudnn_benchmark = False

        if config.perf.allow_tf32:
            args.allow_tf32 = True

#         if config.perf.workers >= 1:
#             args.data_loader_kwargs.num_workers = config.gen.workers

        self.run_desc = desc
        self.args = args

#----------------------------------------------------------------------------
    
    
    def setup_logger(self):
        dnnlib.util.Logger(should_flush=True)
        
        # Pick output directory.
        prev_run_dirs = []
        if os.path.isdir(self.config.gen.output): # outdir
            prev_run_dirs = [x for x in os.listdir(self.config.gen.output) if os.path.isdir(os.path.join(self.config.gen.output, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        self.args.run_dir = os.path.join(self.config.gen.output, f'{cur_run_id:05d}-{self.run_desc}')
        assert not os.path.exists(self.args.run_dir)
        
        print()
        print('Training options:')
        print(json.dumps(self.args, indent=2))
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
        
        
        # Create output directory.
        print('Creating output directory...')
        os.makedirs(self.args.run_dir)

        with open(os.path.join(self.args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(self.args, f, indent=2)
    
    
        dnnlib.util.Logger(file_name=os.path.join(self.args.run_dir, 'log.txt'), file_mode='a', should_flush=True)
        
        if self.config.exp.wandb:
            wandb.init(project=config.exp.project, entity="retir", name=config.exp.name)
            wandb.config = args
        
        # Initialize logs.
        if self.rank == 0:
            print('Initializing logs...')
        stats_collector = training_stats.Collector(regex='.*')
        self.stats_metrics = dict()
        self.stats_jsonl = None
        self.stats_tfevents = None
        if self.rank == 0:
            self.stats_jsonl = open(os.path.join(self.args.run_dir, 'stats.jsonl'), 'wt')
            try:
                import torch.utils.tensorboard as tensorboard
                self.stats_tfevents = tensorboard.SummaryWriter(self.args.run_dir)
            except ImportError as err:
                print('Skipping tfevents export:', err)
        
    
    def init_params(self):
        # Initialize.
        self.wandb_step = 0
        self.start_time = time.time()
        self.device = torch.device('cuda', self.rank)
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
        #self.training_set = dnnlib.util.construct_class_by_name(**self.args.training_set_kwargs) # subclass of training.dataset.Dataset
        self.training_set = datasets[config.dataset](**self.args.training_set_kwargs)
        
        self.training_set_sampler = misc.InfiniteSampler(dataset=self.training_set, rank=self.rank, num_replicas=self.args.num_gpus, seed=random_seed)
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
#         common_kwargs = dict(c_dim=self.training_set.label_dim, img_resolution=self.training_set.resolution, img_channels=self.training_set.num_channels)
#         self.G = dnnlib.util.construct_class_by_name(**self.args.G_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device)
#         self.D = dnnlib.util.construct_class_by_name(**self.args.D_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device)
#         self.G_ema = copy.deepcopy(self.G).eval()
        
        self.G = generators[self.config.generator](**self.args.G_kwargs).train().requires_grad_(False).to(self.device)
        self.D = discriminators[self.config.discriminator](**self.args.D_kwargs).train().requires_grad_(False).to(self.device)
        
        # Resume from existing pickle.
        if (self.args.resume_pkl is not None) and (self.rank == 0):
            print(f'Resuming from "{self.args.resume_pkl}"')
            with dnnlib.util.open_url(self.args.resume_pkl) as f:
                resume_data = legacy.load_network_pkl(f)
            for name, module in [('G', self.G), ('D', self.D)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
                
        # Print network summary tables.
        if self.rank == 0:
            z = torch.empty([self.args.batch_gpu, self.G.z_dim], device=self.device)
            c = torch.empty([self.args.batch_gpu, self.G.c_dim], device=self.device)
            img = misc.print_module_summary(self.G, [z, c])
            misc.print_module_summary(self.D, [img, c])
            
            
    def setup_augmentations(self):
        # Setup augmentation.
        if self.rank == 0:
            print('Setting up augmentation...')
        self.augment_pipe = None
        self.ada_stats = None
        if (self.args.augment_kwargs is not None) and (self.args.augment_p > 0 or self.args.ada_target is not None):
            #self.augment_pipe = dnnlib.util.construct_class_by_name(**self.args.augment_kwargs).train().requires_grad_(False).to(self.device) # subclass of torch.nn.Module
            self.augment_pipe = augmentations[self.config.augmentaion](**self.args.augment_kwargs)
            self.augment_pipe.p.copy_(torch.as_tensor(self.args.augment_p))
            if self.args.ada_target is not None:
                self.ada_stats = training_stats.Collector(regex='Loss/signs/real')
     
    
    def distrib_acrros_gpu(self):
        if self.rank == 0:
            print(f'Distributing across {self.args.num_gpus} GPUs...')
        self.ddp_modules = dict()
        for name, module in [('G', self.G), ('D', self.D), ('augment_pipe', self.augment_pipe)]:
            if (self.args.num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[self.device], broadcast_buffers=False)
                module.requires_grad_(False)
            if name is not None:
                self.ddp_modules[name] = module
                
                
                
    def setup_training_phases(self):
        if self.rank == 0:
            print('Setting up training phases...')
        #self.loss = dnnlib.util.construct_class_by_name(device=self.device, **self.args.ddp_modules, **self.args.loss_kwargs) # subclass of training.loss.Loss
        self.loss = losses_arch[self.config.loss_arch](device=self.device,
                                                       loss=self.config.loss,
                                                       gen_regs = [(key, config.gen_regs_all[key]) for key in config.gen.gen_regs],
                                                       disc_regs = [(key, config.disc_regs_all[key]) for key in config.gen.disc_regs],
                                                       **self.args.ddp_modules,
                                                       **self.args.loss_kwargs)
        self.phases = []
        for name, module, opt_kwargs, reg_interval in [('G', self.G, self.args.G_opt_kwargs, self.args.G_reg_interval), ('D', self.D, self.args.D_opt_kwargs, self.args.D_reg_interval)]:
            if reg_interval is None:
                #self.opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                self.opt = optimizers[self.config.optimizers[name]](params=module.parameters(), **opt_kwargs)
                self.phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=self.opt, interval=1)]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                #self.opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                self.opt = optimizers[self.config.optimizer](params=module.parameters(), **opt_kwargs)
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
        for name, module in [('G', self.G), ('D', self.D), ('augment_pipe', self.augment_pipe)]:
            if module is not None:
                if self.args.num_gpus > 1:
                    misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            snapshot_data[name] = module
            del module # conserve memory
        snapshot_pkl = os.path.join(self.args.run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
        if self.rank == 0:
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
        return snapshot_data, snapshot_pkl
    
    
    def evaluate_metrics(self, snapshot_data, snapshot_pkl):
        if self.rank == 0:
            print('Evaluating metrics...')
        for metric in self.args.metrics:
            result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G'],
                dataset_kwargs=self.args.training_set_kwargs, num_gpus=self.args.num_gpus, rank=self.rank, device=self.device)
            if self.config.exp.wandb:
                if metric == 'fid50k_full':
                    wandb.log({'FID': result_dict['results']['fid50k_full']}, step=self.wandb_step, commit=False)
                if metric == 'is50k':
                    wandb.log({'IS': result_dict['results']['is50k_mean']},step=self.wandb_step, commit=False)
            if rank == 0:
                metric_main.report_metric(result_dict, run_dir=self.args.run_dir, snapshot_pkl=snapshot_pkl)
            self.stats_metrics.update(result_dict.results)

    
    
    def training_loop(self):
        if rank == 0:
            print(f'Training for {self.args.total_kimg} kimg...')
            print()
        cur_nimg = 0
        cur_tick = 0
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - self.start_time
        batch_idx = 0
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
                    phase.start_event.record(torch.cuda.current_stream(device))
                phase.opt.zero_grad(set_to_none=True)
                phase.module.requires_grad_(True)

                # Accumulate gradients over multiple rounds.
                for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                    sync = (round_idx == self.args.batch_size // (self.args.batch_gpu * self.args.num_gpus) - 1)
                    gain = phase.interval
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)
                    

                # Update weights.
                phase.module.requires_grad_(False)
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    for param in phase.module.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    phase.opt.step()
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

            # Update state.
            cur_nimg += self.argsbatch_size
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
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"time left {dnnlib.util.format_time(int((tick_end_time - start_time) / cur_nimg * (self.args.total_kimg * 1000 - cur_nimg))):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(self.device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            fields += [f"augment {training_stats.report0('Progress/augment', float(self.augment_pipe.p.cpu()) if self.augment_pipe is not None else 0):.3f}"]
            training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
            if rank == 0:
                print(' '.join(fields))

            # Check for abort.
            if (not done) and (self.abort_fn is not None) and self.abort_fn():
                done = True
                if rank == 0:
                    print()
                    print('Aborting...')

            # Save image snapshot.
            if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                images = torch.cat([self.G(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
                image = PIL.Image.open(os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'))
                if self.config.exp.wandb:
                    wandb.log({"Generated images": wandb.Image(image)}, step=wandb_step, commit=False)


            # Save network snapshot.
            snapshot_data = None
            if (self.args.network_snapshot_ticks is not None) and (done or cur_tick % self.args.network_snapshot_ticks == 0):
                snapshot_data, snapshot_pkl = self.save_snapshot(cur_nimg)

            # Evaluate metrics.
            if (snapshot_data is not None) and (len(self.args.metrics) > 0):
                evaluate_metrics(snapshot_data, snapshot_pkl)
            del snapshot_data # conserve memory

            # Collect statistics.
            for phase in self.phases:
                value = []
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
            stats_collector.update()
            stats_dict = stats_collector.as_dict()

            # Update logs.
            if self.config.exp.wandb:
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
                for name, value in stats_metrics.items():
                    self.stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                self.stats_tfevents.flush()
            if self.progress_fn is not None:
                self.progress_fn(cur_nimg // 1000, self.args.total_kimg)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
            if done:
                break

                

        