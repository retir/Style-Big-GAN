import utils
from dataclasses import dataclass, field

augpipe_specs = utils.ClassRegistry()


@augpipe_specs.add_to_registry("bgc")
@dataclass
class BGSada:
    xflip: float = 1
    rotate90: float = 1
    xint: float = 1
    scale: float = 1
    rotate: float = 1
    aniso: float = 1
    xfrac: float = 1
    brightness: float = 1
    contrast: float = 1
    lumaflip: float = 1
    hue: float = 1
    saturation: float = 1
        
        