import utils
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from types import MappingProxyType
from omegaconf import OmegaConf, MISSING
from frozendict import frozendict


first_coll = utils.ClassRegistry()
second_coll = utils.ClassRegistry()


@dataclass
class D:
    val1: float = 0.1
    val2: str = 'ass'


@first_coll.add_to_registry("a")
@dataclass
class A:
    list_field: Tuple = ('val1', 'val2')#field(default_factory=lambda: ('val1', 'val2'))
    dict_field: D =  D()
        
        
B = first_coll.make_dataclass_from_args("B")
second_coll.add_to_registry("b")(B)

# @second_coll.add_to_registry("c")
# @dataclass
# class C:
#     list_field: List[str] = field(default_factory=lambda: ['val1', 'val2'])





Args = second_coll.make_dataclass_from_classes("Args")
config = OmegaConf.structured(Args)

