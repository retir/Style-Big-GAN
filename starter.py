import os
import torch
import arguments
from launch import launcher
import json
from omegaconf import OmegaConf

def main(config):
    print('START MAIN')
    launcher(config)
    print('COMPLETE')


if __name__ == "__main__":
    # limit CPU usage
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OPENMP_NUM_THREADS"] = "1"
    torch.multiprocessing.set_start_method("spawn")

    config = arguments.load_config()
    print('CONFIG LOADED')
    #print(type(OmegaConf.to_container(config)))
    main(config)