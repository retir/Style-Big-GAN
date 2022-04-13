import utils
from dataclasses import dataclass, field
from stylegan2ada.dnnlib import EasyDict


gens_args = utils.ClassRegistry()
discs_args = utils.ClassRegistry()


@dataclass
class Mappingkwargs:
    num_layers: int = 8

@gens_args.add_to_registry("sg2_classic")
@dataclass
class SG2classic_gen_args:
    z_dim: int = 128
    w_dim: int = 128
    mapping_kwargs: Mappingkwargs = Mappingkwargs()
#     mapping_kwargs: EasyDict = field(default_factory=lambda: EasyDict({                    
#         'num_layers': 8  # смотри base cfg
#     }))
#     synthesis_kwargs: EasyDict = field(default_factory=lambda: EasyDict({
#         'channel_base': 32768,  # смотри base cfg
#         'channel_max':  512,
#         'num_fp16_res':  4,
#         'conv_clamp':  256,
#         'fp16_channels_last': False
#     }))

        
@discs_args.add_to_registry("sg2_classic")
@dataclass       
class SG2classic_dis_args:
#     block_kwargs: EasyDict = field(default_factory=lambda: EasyDict({
#         'fp16_channels_last': False
#     }))
#     mapping_kwargs: EasyDict = field(default_factory=lambda: EasyDict({}))
#     epilogue_kwargs: EasyDict = field(default_factory=lambda: EasyDict({
#         'mbstd_group_size': 4  # смотри base cfg
#     }))
    channel_base: int = 32768  # смотри base cfg
    channel_max: int = 512
    num_fp16_res: int = 4

    