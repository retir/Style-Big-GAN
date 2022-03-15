import librosa
import torch.utils.data
import torch.distributions
import numpy as np
import random
import math
import os
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
import utils
import scipy
from dataclasses import dataclass, field
from typing_extensions import Literal
from collections import defaultdict 


models = utils.ClassRegistry()

@dataclass
class Base_model:
    dataset: str = 'cifar10'
    total_steps: int = 100000
    lr_G: float = 2e-4
    lr_D: float = 2e-4
    batch_size: int = 128
    beta_0: float = 0.5
    beta_1: float = 0.9
    n_dis: int = 5
    z_dim: int = 128
    seed: int = 0


@models.add_to_registry("dcgan")
@dataclass
class ExperimentArgs(Base_model):
    arch: str = 'cnn32_dcgan'
    total_steps: int = 50000
    n_dis: int = 1
    z_dim: int = 100
    loss: str = 'bce'



@models.add_to_registry("sngan")
@dataclass
class ExperimentArgs(Base_model):
    arch: str = 'res32_sngan'
    batch_size: int = 64
    beta_0: float = 0.0
    loss: str = 'hinge'


@models.add_to_registry("wgan")
@dataclass
class ExperimentArgs(Base_model):
    arch: str = 'cnn_32_wgan'
    loss: str = 'was'
    c: float = 0.1


@models.add_to_registry("wgangp")
@dataclass
class ExperimentArgs(Base_model):
    arch: str = 'cnn_32_wgan'
    batch_size: int = 64
    loss: str = 'was'
    alpha: float = 10
    beta_0: float = 0.0






