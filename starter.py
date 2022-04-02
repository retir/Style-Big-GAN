import os
import torch
import arguments
import json
from omegaconf import OmegaConf
from trainers.gan_collections_trainer import gan_collections_launch
from trainers.sg2trainer import sg2_launch

def main(config):
    print('START MAIN')
    launcher(config)
    print('COMPLETE')


if __name__ == "__main__":
    config = arguments.load_config()
    print('CONFIG LOADED')
    
    if config.exp.trainer == 'stylegan':
        sg2_launch(config)
    elif config.exp.trainer == 'gan-collections':
        gan_collections_launch(config)
    else:
        raise ValueError('Wrong trainer')
