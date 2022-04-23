import utils
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

gen_regs_args = utils.ClassRegistry()
disc_regs_args = utils.ClassRegistry()


@gen_regs_args.add_to_registry("ppl")
@dataclass
class PPL_args:
    pl_batch_shrink: float = 0.0025
    #pl_decay: List[str] = field(default_factory=lambda: ['pepega', 'pep'])
    #metrics: List[str] = field(default_factory=list)
    pl_weight: float = 1e-8
        
        
@disc_regs_args.add_to_registry("r1")
@dataclass
class R1_args:
    r1_gamma: float = 1. # spec
        
        
@disc_regs_args.add_to_registry("grad_pen")
@dataclass
class GradPen_args:
    alpha: float = 10. # spec

    