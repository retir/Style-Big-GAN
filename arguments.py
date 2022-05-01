import os
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
import utils

from train_parts_args.augmentaions_agrs import augpipe_specs
from train_parts_args.datasets_args import datasets_args, dataloaders_args
from train_parts_args.gens_and_discs_args import gens_args, discs_args
from train_parts_args.losses_arch_args import losses_arch_args
from train_parts_args.optim_args import optim_gen_args, optim_disc_args
from train_parts_args.regs_args import gen_regs_args, disc_regs_args


args = utils.ClassRegistry()


@args.add_to_registry("exp")
@dataclass
class ExperimentArgs:
    config_dir: str = MISSING
    config: str = MISSING
    name: str = MISSING
    project: str = "gan-collections"
    notes: str = "empty notes"
    dry_run: bool = False
    trainer: str = 'base'

        
@args.add_to_registry("data")
@dataclass
class Dataset:
    dataset: str = 'image_folder'
    dataloader: str = 'basic'
    dataset_path: str = './data'
    cond: bool = False
    subset: int = 0 
    mirror: bool = False
    

@args.add_to_registry("log")
@dataclass
class LogArgs:
    snap: int = 50
    output: str = './outputs'
    metrics: List[str] = field(default_factory=lambda: ['fid50k_full', 'is50k'])
    kimg_per_tick: int = 4
    wandb: bool = True



@args.add_to_registry("gen")
@dataclass
class GenArgs:
    kimg: int = -1
    batch: int = -1
    batch_gpu: int = 32
    seed: int = 0
    generator: str = 'sg2_classic'
    discriminator: str = 'sg2_classic'
    optim_gen: str = 'adam'
    optim_disc: str = 'adam'
    gen_regs: List[str] = field(default_factory=lambda: [])
    disc_regs: List[str] = field(default_factory=lambda: [])
    loss_arch: str = 'sg2'
    loss: str = 'softplus'
    g_reg_interval: int = 16
    d_reg_interval: int = 4
    n_dis: int = 1

        
@args.add_to_registry("perf")
@dataclass
class Performance:
    fp32: bool = False
    nhwc: bool = False
    allow_tf32: bool = False
    nobench: bool = False
    gpus: int = 1
        
        
@args.add_to_registry("ema")
@dataclass
class EmaArgs:
    use_ema: bool = True
    kimg: int = 20
    ramp: float = -1 
        

@args.add_to_registry("aug")
@dataclass
class Augmentations:
    aug: str = 'ada'
    aug_type: str = 'sg2_ada'
    p: float = -1 
    target: float = -1
    augpipe: str = 'bgc'
        
        
@args.add_to_registry("trans")
@dataclass
class TransferLearning:
    resume: str = 'noresume'
    resume_url: str = ''
    freezed: int = -1
        
DatasetArgs = datasets_args.make_dataclass_from_args("DatasetArgs")
args.add_to_registry("datasets_args")(DatasetArgs)

DataloaderArgs = dataloaders_args.make_dataclass_from_args("DataloaderArgs")
args.add_to_registry("dataloaders_args")(DataloaderArgs)

GensArgs = gens_args.make_dataclass_from_args("GensArgs")
args.add_to_registry("gens_args")(GensArgs)

DiscsArgs = discs_args.make_dataclass_from_args("DiscsArgs")
args.add_to_registry("discs_args")(DiscsArgs)

OptimGenArgs = optim_gen_args.make_dataclass_from_args("OptimGenArgs")
args.add_to_registry("optim_gen_args")(OptimGenArgs)

OptimDiscArgs = optim_disc_args.make_dataclass_from_args("OptimDiscArgs")
args.add_to_registry("optim_disc_args")(OptimDiscArgs)

LossesArchArgs = losses_arch_args.make_dataclass_from_args("LossesArchArgs")
args.add_to_registry("losses_arch_args")(LossesArchArgs)

AugpipeArgs = augpipe_specs.make_dataclass_from_args("AugpipeArgs")
args.add_to_registry("augpipe_specs")(AugpipeArgs)

GenRegsArgs = gen_regs_args.make_dataclass_from_args("GenRegsArgs")
args.add_to_registry("gen_regs_all")(GenRegsArgs)

DiscRegsArgs = disc_regs_args.make_dataclass_from_args("DiscRegsArgs")
args.add_to_registry("disc_regs_all")(DiscRegsArgs)


Args = args.make_dataclass_from_classes("Args")


def load_config():
    config = OmegaConf.structured(Args)

    conf_cli = OmegaConf.from_cli()
    config.exp.config = conf_cli.exp.config
    config.exp.config_dir = conf_cli.exp.config_dir

    config_path = os.path.join(config.exp.config_dir, config.exp.config)
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, conf_file)

    config = OmegaConf.merge(config, conf_cli)

    return config