import os
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
import utils
import models


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
    num_images: int = 50000


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