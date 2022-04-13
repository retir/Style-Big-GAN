import os
import torch
import arguments2 as arguments
import json
from omegaconf import OmegaConf
from trainerss import trainers

def main():
    print('Loading config...')
    config = arguments.load_config()
    print('Config loaded')
    
    trainer = trainers[config.exp.trainer](config)
    #trainer.config = config
    trainer.setup_arguments()
    # Dry run?
    if trainer.config.exp.dry_run:
        print('Dry run; exiting.')
        return
#     trainer.setup_logger()
    
#     print('Launching multiprocesses...')
#     torch.multiprocessing.set_start_method('spawn')
#     with tempfile.TemporaryDirectory() as temp_dir:
#         if config.num_gpus == 1:
#             multiprocesses_main(rank=0, trainer=trainer)
#         else:
#             torch.multiprocessing.spawn(fn=multiprocesses_main, args=(trainer), nprocs=config.num_gpus)

def multiprocesses_main(rank, trainer):
    trainer.rank = rank
    
    trainer.init_params()
    trainer.setup_dataset()
    trainer.setup_augmentations()
    trainer.distrib_acrros_gpu()
    trainer.setup_training_phases()
    
    trainer.training_loop()
    
    


if __name__ == "__main__":
    main()
    
    