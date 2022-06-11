import utils
import torch

optimizers= utils.ClassRegistry()


@optimizers.add_to_registry("adam")
class Adam(torch.optim.Adam):
    def __init__(self, params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, **args):
        assert params is not None
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, **args)