# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.init as init
from torch.nn.utils.spectral_norm import spectral_norm
import math
import utils
import functools
from biggan import layers
from stylegan2ada.torch_utils import misc
from stylegan2ada.torch_utils import persistence
from stylegan2ada.torch_utils.ops import conv2d_resample
from stylegan2ada.torch_utils.ops import upfirdn2d
from stylegan2ada.torch_utils.ops import bias_act
from stylegan2ada.torch_utils.ops import fma
from biggan.layers import Attention

#----------------------------------------------------------------------------

generators = utils.ClassRegistry()

#----------------------------------------------------------------------------

# Style GAN 2
#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        attention           = False,        # Use attention in the end of the block
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.attention = Attention(out_channels) if attention else None

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        #self.conv0_5 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution, # DEL
        #    conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        #self.num_conv += 1
        
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            #x = self.conv0_5(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs) # DEL
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            #x = self.conv0_5(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs) # DEL
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            #x = self.conv0_5(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs) # DEL
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        if self.attention:
            x_type = x.type()
            x = self.attention(x.to(torch.float32)).type(x_type)
            
        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y
        
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        attentions      = [],
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, attention=res in attentions, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)

        return img

#----------------------------------------------------------------------------

@generators.add_to_registry("sg2_classic")
@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        attentions          = [],
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, attentions=attentions,**synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img

# DCGAN
#----------------------------------------------------------------------------

class Generator_dcgan(torch.nn.Module):
    def __init__(self, z_dim, M):
        super(Generator_dcgan, self).__init__()
        self.z_dim = z_dim
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.z_dim, 1024, M, 1, 0, bias=False),  # 4, 4
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, z, c, noise_mode=None):
        return self.main(z.view(-1, self.z_dim, 1, 1))


@generators.add_to_registry("cnn32_dcgan")
class Generator32_dcgan(Generator_dcgan):
    def __init__(self, z_dim, c_dim, img_resolution, *args, **kwargs):
        super().__init__(z_dim, M=2)
        self.c_dim = c_dim
        self.img_resolution = img_resolution


@generators.add_to_registry("cnn48_dcgan")
class Generator48_dcgan(Generator_dcgan):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=4)
        
# SNGAN
#----------------------------------------------------------------------------

class ResGenBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


@generators.add_to_registry("res32_sngan")
class ResGenerator32_SNGAN(torch.nn.Module):
    def __init__(self, z_dim, c_dim, img_resolution, *args, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.linear = torch.nn.Linear(z_dim, 4 * 4 * 256)

        self.blocks = torch.nn.Sequential(
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
        )
        self.output = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 3, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, c, noise_mode='random'):
        z = self.linear(z)
        z = z.view(-1, 256, 4, 4)
        return self.output(self.blocks(z))
    

# WGAN  
#----------------------------------------------------------------------------

@generators.add_to_registry("res32_wgan")
class ResGenerator32_WGAN(torch.nn.Module):
    def __init__(self, z_dim, c_dim, img_resolution, *args, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.linear = torch.nn.Linear(z_dim, 4 * 4 * 256)

        self.blocks = torch.nn.Sequential(
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
        )
        self.output = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 3, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, c, noise_mode='random'):
        z = self.linear(z)
        z = z.view(-1, 256, 4, 4)
        return self.output(self.blocks(z))
    

# Big GAN   
#-----------------------------------------------------------------------------------

def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
    arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
    arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
    arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
    arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

    return arch


@generators.add_to_registry("big_gan")
class BigGAnGenerator(torch.nn.Module):
    def __init__(self, G_ch=64, z_dim=128, c_dim=10, bottom_width=4, img_resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=10,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=torch.nn.ReLU(inplace=False),
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho',
               G_param='SN', norm_style='bn',
               **kwargs):
        super(BigGAnGenerator, self).__init__()
        self.c_dim = c_dim
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.z_dim = z_dim
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.img_resolution = img_resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else z_dim
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[img_resolution]

        # If using hierarchical latents, adjust z
        if self.hier:
          # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.z_dim // self.num_slots)
          # Recalculate latent dimensionality for even splitting into chunks
            self.z_dim = self.z_chunk_size *  self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                              kernel_size=3, padding=1,
                              num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                              eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                              num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                              eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(torch.nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = torch.nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = torch.nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                     else self.which_embedding)
        self.which_bn = functools.partial(layers.ccbn,
                              which_linear=bn_linear,
                              cross_replica=self.cross_replica,
                              mybn=self.mybn,
                              input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                          else self.n_classes),
                              norm_style=self.norm_style,
                              eps=self.BN_eps)


        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                        else layers.identity())
        # First linear layer
        self.linear = self.which_linear(self.z_dim // self.num_slots,
                                        self.arch['in_channels'][0] * (self.bottom_width **2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                 out_channels=self.arch['out_channels'][index],
                                 which_conv=self.which_conv,
                                 which_bn=self.which_bn,
                                 activation=self.activation,
                                 upsample=(functools.partial(torch.nn.functional.interpolate, scale_factor=2)
                                           if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = torch.nn.ModuleList([torch.nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = torch.nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                        self.activation,
                                        self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        
        self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, torch.nn.Conv2d) 
              or isinstance(module, torch.nn.Linear) 
              or isinstance(module, torch.nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

        
    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, c, noise_mode='random'):
    # If hierarchical, concatenate zs and ys
        y = torch.argmax(c, dim=1)

        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
          # Second inner loop in case block has multiple layers
          for block in blocklist:
            h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


