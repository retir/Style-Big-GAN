import os
import torch
import arguments2 as arguments
import json
import tempfile
import wandb
from omegaconf import OmegaConf
from trainerss import trainers
from stylegan2ada.torch_utils import training_stats
from stylegan2ada.torch_utils import custom_ops

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
    
    print('Launching multiprocesses...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if config.gen.gpus == 1:
            multiprocesses_main(rank=0, trainer=trainer, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=multiprocesses_main, args=(trainer, temp_dir), nprocs=config.num_gpus)

def multiprocesses_main(rank, trainer, temp_dir):
    trainer.rank = rank
    
    trainer.setup_logger()
#     if trainer.config.exp.wandb:
#             wandb.init(project=trainer.config.exp.project, entity="retir", name=trainer.config.exp.name)
    
    # Init torch.distributed.
    if trainer.args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=trainer.args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=trainer.args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if trainer.args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'
    
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
    
    