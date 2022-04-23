import utils
import torch

optimizers= utils.ClassRegistry()


@optimizers.add_to_registry("adam")
class Adam(torch.optim.Adam):
    def __init__(self, **args):
        super().__init__(**args)