import utils
from dataclasses import dataclass, field

datasets_args = utils.ClassRegistry()
dataloaders_args = utils.ClassRegistry()


@datasets_args.add_to_registry("image_folder")
@dataclass
class SG2classic_gen_args:
    use_labels: bool = True
    xflip: bool = False
    max_size: int = -1  # None
        

        
@dataloaders_args.add_to_registry("basic")
@dataclass
class BasicDataloader:
    pin_memory: bool = True
    num_workers: int = 3
    prefetch_factor: int = 2   


        

    