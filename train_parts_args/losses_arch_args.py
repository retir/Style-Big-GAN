import utils
from dataclasses import dataclass, field

losses_arch_args = utils.ClassRegistry()

@losses_arch_args.add_to_registry("base")
@dataclass
class Base_lose:
    pass

        
@losses_arch_args.add_to_registry("sg2")
@dataclass       
class SG2classic_dis_args:
    style_mixing_prob: float = 0.9

    