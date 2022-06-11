import utils
import torch
from dataclasses import dataclass, field

dataloaders = utils.ClassRegistry()


@dataloaders.add_to_registry("basic")
class BasicDataloader(torch.utils.data.DataLoader):
    def __init__(self, pin_memory=True, num_workers=3, prefetch_factor=2, **args):
        args.update({'pin_memory': pin_memory, 'num_workers': num_workers, 'prefetch_factor': prefetch_factor})
        super().__init__(**args)

