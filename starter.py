import os
import torch
import arguments
import json
import tempfile
import wandb
from omegaconf import OmegaConf
from train_parts.trainers import trainers
from stylegan2ada.torch_utils import training_stats
from stylegan2ada.torch_utils import custom_ops

def main():
    print('Loading config...')
    config = arguments.load_config()
    print('Config loaded')
    
    trainer = trainers[config.exp.trainer](config)
    trainer.setup_arguments()
    
    if trainer.config.exp.dry_run:
        print('Dry run; exiting.')
        return
    
    print('Launching multiprocesses...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if config.perf.gpus == 1:
            multiprocesses_main(rank=0, trainer=trainer, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=multiprocesses_main, args=(trainer, temp_dir), nprocs=config.num_gpus)

def multiprocesses_main(rank, trainer, temp_dir):
    trainer.rank = rank
    
    trainer.setup_logs()
    trainer.distribute_torch(temp_dir)
    trainer.init_params()
    trainer.setup_dataset()
    trainer.setup_networks()
    trainer.setup_augmentations()
    trainer.distrib_acrros_gpu()
    trainer.setup_training_phases()
    trainer.export_sample_images()
    
    trainer.training_loop()
    

if __name__ == "__main__":
    main()
    
    