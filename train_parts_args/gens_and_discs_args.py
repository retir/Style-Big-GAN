import utils
from dataclasses import dataclass, field
from stylegan2ada.dnnlib import EasyDict


gens_args = utils.ClassRegistry()
discs_args = utils.ClassRegistry()


@dataclass
class Mappingkwargs:
    num_layers: int = 8
        

@dataclass
class Synthesiskwargs:
    channel_base: int = 32768  # смотри base cfg
    channel_max: int = 512
    num_fp16_res: int = 4
    conv_clamp: int = 256
    fp16_channels_last: bool = False
    

@gens_args.add_to_registry("sg2_classic")
@dataclass
class SG2classic_gen_args:
    z_dim: int = 128
    w_dim: int = 128
    mapping_kwargs: Mappingkwargs = Mappingkwargs()
    synthesis_kwargs: Synthesiskwargs = Synthesiskwargs()
        

@dataclass
class Blockkwargs:
    fp16_channels_last: bool = False
        
        
        
@dataclass
class Epiloguekwargs:
    mbstd_group_size: int = 4


        
@discs_args.add_to_registry("sg2_classic")
@dataclass       
class SG2classic_dis_args:
    block_kwargs: Blockkwargs = Blockkwargs()
    mapping_kwargs: Mappingkwargs = Mappingkwargs()
    epilogue_kwargs: Epiloguekwargs = Epiloguekwargs()
    conv_clamp: int = 256
    channel_base: int = 32768 
    channel_max: int = 512
    num_fp16_res: int = 4
    architecture: str = "resnet"
        
        
        
@gens_args.add_to_registry("cnn32_dcgan")
@dataclass
class DcganCNN32_gen:
    z_dim: int = 100
        

@discs_args.add_to_registry("cnn32_dcgan")
@dataclass
class DcganCNN32_dis:
    z_dim: int = 100
        
        
@gens_args.add_to_registry("res32_sngan")
@dataclass
class res32SNGAN_gen:
    z_dim: int = 128
        
@discs_args.add_to_registry("res32_sngan")
@dataclass
class res32SNGAN_dis:
    z_dim: int = 128
        
        

@gens_args.add_to_registry("res32_wgan")
@dataclass
class res32SNGAN_gen:
    z_dim: int = 128
        
@discs_args.add_to_registry("res32_wgan")
@dataclass
class res32SNGAN_dis:
    z_dim: int = 128
        
        
@discs_args.add_to_registry("big_gan")
@dataclass
class BigGan_dis:
    D_ch: int = 64
    D_wide: bool = True
    resolution: int = 128
    D_kernal_size: int = 3
    D_attn: str = '64'
    n_classes: int = 1000
    num_D_SVs: int = 1
    num_D_SV_itrs: int = 1
    D_activations: str = 'relu'
    SN_eps: float = 1e-12
    output_dim: int = 1
    D_mixed_precision: bool = False
    D_fp16: bool = False
    D_init: str = 'ortho'
    skip_init: bool = False
    D_params: str = 'SN'
  

@gens_args.add_to_registry("big_gan")
@dataclass
class BigGan_gen:
    G_ch: int = 64
    dim_z: int = 128
    bottom_width: int = 4
    resolution: int = 128
    G_kernal_size: int = 3
    G_attn: str = '64'
    n_classes: int = 1000
    num_G_SVs: int = 1
    num_G_SV_itrs: int = 1
    G_shared: bool = True
    shared_dim: int = 0
    hier: bool = False
    cross_replica: bool = False
    mybn: bool = False
    G_activations: str = 'relu'
    BN_eps: float = 1e-5
    SN_eps: float = 1e-12
    G_mixed_precision: bool = False
    G_fp16: bool = False
    G_init: str = 'ortho'
    skip_init: bool = False
    G_params: str = 'SN'
    norm_style: str = 'bn'

    