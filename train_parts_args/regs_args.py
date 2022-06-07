import utils
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

gen_regs_args = utils.ClassRegistry()
disc_regs_args = utils.ClassRegistry()


@gen_regs_args.add_to_registry("ppl")
@dataclass
class PPL_args:
    pl_batch_shrink: float = 2 #0.0025
    pl_weight: float = 2 #1e-8
    pl_decay: float = 0.01
        
        
@disc_regs_args.add_to_registry("r1")
@dataclass
class R1_args:
    r1_gamma: float = 10. # spec
        
        
@disc_regs_args.add_to_registry("grad_pen")
@dataclass
class GradPen_args:
    alpha: float = 10. # spec

    