import utils
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

optim_gen_args = utils.ClassRegistry()
optim_disc_args = utils.ClassRegistry()
    
    
@optim_disc_args.add_to_registry("adam")
@dataclass
class Adam_args_disc:
    lr: float = 0.002 # spec
    betas: Tuple = (0, 0.99)
    eps: float = 1e-8
        
@optim_gen_args.add_to_registry("adam")
@dataclass
class Adam_args_disc:
    lr: float = 0.002 # spec
    betas: Tuple = (0, 0.99)
    eps: float = 1e-8

        


    