import os
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
import utils
import gan_collections.models as models


args = utils.ClassRegistry()

@args.add_to_registry("exp")
@dataclass
class ExperimentArgs:
    config_dir: str = MISSING
    config: str = MISSING
    name: str = MISSING
    project: str = "gan-collections"
    notes: str = "empty notes"
    pretrain: bool = False
    pretrain_path: str = './logs'
    base_cfg: str = 'auto'
    dry_run: bool = False
    trainer: str = 'stylegan'

        
@args.add_to_registry("data")
@dataclass
class Dataset:
    dataset: str = 'cifar10'
    dataset_path: str = './data'
    snap: int = 50
    cond: bool = False
    subset: int = 0 # default = all?
    mirror: bool = False
    

@args.add_to_registry("log")
@dataclass
class LogArgs:
    eval_step: int = 5000
    sample_step: int = 500
    sample_size: int = 64
    logdir: str = './logs'
    record: bool = True
    fid_cache: str = './stats/cifar10.train.npz'



@args.add_to_registry("gen")
@dataclass
class GenArgs:
    model: str = MISSING
    generate: bool = False
    pretrain: str = MISSING
    output: str = './outputs'
    metrics: List[str] = field(default_factory=lambda: ['fid50k_full', 'is50k'])
    gamma: float = -1
    kimg: int = -1
    batch: int = -1 # Поправить +
    seed: int = 0
        
    fp32: bool = False
    nhwc: bool = False
    allow_tf32: bool = False
    nobench: bool = False
    gpus: int = 1
    workers: int = 3
        

@args.add_to_registry("aug")
@dataclass
class Augmentations:
    aug: str = 'ada'
    p: float = -1 # default = ?
    target: float = -1 # default = ?
    augpipe: str = 'bgc'
        
        
@args.add_to_registry("trans")
@dataclass
class TransferLearning:
    resume: str = 'noresume'
    freezed: int = -1
        
ModelsArgs = models.models.make_dataclass_from_args("ModelsArgs")
args.add_to_registry("model_params")(ModelsArgs)


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