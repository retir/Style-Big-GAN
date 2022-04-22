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
import json

from stylegan2ada.training import training_loop
from stylegan2ada.metrics import metric_main
from stylegan2ada.torch_utils import training_stats
from stylegan2ada.torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------
import os
import torch
import json
from omegaconf import OmegaConf

def setup_training_loop_kwargs(config):
    if config.trans.resume == 'from_data':
        with open(os.path.join(config.trans.resume_dir, config.trans.args_name)) as json_data:
            args = dnnlib.EasyDict(json.load(json_data))
        args.resume_params = dnnlib.EasyDict(config.trans)
        desc = args.desc
        return desc, args
    
    args = dnnlib.EasyDict()
    
    args.start_options = dnnlib.EasyDict({'cur_nimg' : 0,
        'cur_tick' : 0,
        'batch_idx' : 0,
         'wandb_step' : 0})

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------
   
    assert isinstance(config.gen.gpus, int)
    if not (config.gen.gpus >= 1 and config.gen.gpus & (config.gen.gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = config.gen.gpus
    
    args.wandb_use = config.exp.wandb


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
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=config.data.dataset_path, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
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

    desc += f'-{config.exp.base_cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert config.exp.base_cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[config.exp.base_cfg])
    if config.exp.base_cfg == 'auto':
        print('AUTO SETTINGS')
        desc += f'{config.gen.gpus:d}'
        spec.ref_gpus = config.gen.gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(config.gen.gpus * min(4096 // res, 32), 64), config.gen.gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // config.gen.gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if config.exp.base_cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    if config.gen.gamma >= 0:
        desc += f'-gamma{config.gen.gamma:g}'
        args.loss_kwargs.r1_gamma = config.gen.gamma

    if config.gen.kimg >= 1:
        desc += f'-kimg{config.gen.kimg:d}'
        args.total_kimg = config.gen.kimg

    if config.gen.batch > 0:
        if not (config.gen.batch >= 1 and config.gen.batch % config.gen.gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{config.gen.batch}'
        args.batch_size = config.gen.batch
        args.batch_gpu = config.gen.batch // config.gen.gpus
        args.batch_gpu = 32

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

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert config.aug.augpipe in augpipe_specs
    if config.aug.aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[config.aug.augpipe])

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
        args.resume_pkl = config.trans.resume # custom path or url

    if config.trans.resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if config.trans.freezed >= 0:
        desc += f'-freezed{config.trans.freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = config.trans.freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if config.gen.fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if config.gen.nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if config.gen.nobench:
        args.cudnn_benchmark = False

    if config.gen.allow_tf32:
        args.allow_tf32 = True

    if config.gen.workers >= 1:
        args.data_loader_kwargs.num_workers = config.gen.workers
        
    args.desc = desc

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir, config):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    
    if args.wandb_use:
        if config.trans.resume == 'noresume':
            wandb_id = wandb.util.generate_id()
            args.wandb_id = wandb_id
            args.wandb_project = config.exp.project
            wandb.init(id=wandb_id, resume='allow', project=config.exp.project, entity="retir", name=config.exp.name)
            wandb.config = args
        else:
            wandb.init(id=args.wandb_id, resume=True, project=args.wandb_project, entity="retir")

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none' #  COMMENT

    # Execute training loop.
    training_loop.training_loop(rank=rank, args=args, **args) # LOOOOP

#----------------------------------------------------------------------------

# class CommaSeparatedList(click.ParamType):
#     name = 'list'

#     def convert(self, value, param, ctx):
#         _ = param, ctx
#         if value is None or value.lower() == 'none' or value == '':
#             return []
#         return value.split(',')

#----------------------------------------------------------------------------

def sg2_launch(config):
    
    dnnlib.util.Logger(should_flush=True)
    
#     config = arguments.load_config()
#     print('CONFIG LOADED')
    
    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(config)
    except UserError as err:
        raise err

    # Pick output directory.
    if config.trans.resume != 'from_data':
        prev_run_dirs = []
        if os.path.isdir(config.gen.output): # outdir
            prev_run_dirs = [x for x in os.listdir(config.gen.output) if os.path.isdir(os.path.join(config.gen.output, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        args.run_dir = os.path.join(config.gen.output, f'{cur_run_id:05d}-{run_desc}')
        assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs["path"]}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs["max_size"]}')
    print(f'Image resolution:   {args.training_set_kwargs["resolution"]}')
    print(f'Conditional model:  {args.training_set_kwargs["use_labels"]}')
    print(f'Dataset x-flips:    {args.training_set_kwargs["xflip"]}')
    print()

    # Dry run?
    if config.exp.dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    if config.trans.resume != 'from_data':
        print('Creating output directory...')
        os.makedirs(args.run_dir)
        with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(args, f, indent=2)
    
    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir, config=config)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir, config), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

# if __name__ == "__main__":
#     print('START MAIN')
#     main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
