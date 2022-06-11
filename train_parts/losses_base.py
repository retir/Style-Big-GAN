# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from stylegan2ada.torch_utils import training_stats
from stylegan2ada.torch_utils import misc
from stylegan2ada.torch_utils.ops import conv2d_gradfix
import stylegan2ada.dnnlib as dnnlib
import utils
from train_parts.regularizations import generator_regs
from train_parts.regularizations import discriminator_regs
from train_parts.losses import losses

losses_arch = utils.ClassRegistry()

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------


class LossBase:
    def __init__(self, device, gen_regs, dis_regs, G, D, loss, augment_pipe=None):
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.gen_regs = [generator_regs[reg](**args) for reg, args in gen_regs] if len(gen_regs) > 0 else None
        self.dis_regs = [discriminator_regs[reg](**args) for reg, args in dis_regs] if len(dis_regs) > 0 else None
        self.loss = losses[loss]()
    
    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G, sync):
            img = self.G(z, c)
        return img

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits
        
    def do_Gmain(self, real_img, real_c, gen_z, gen_c, sync, gain):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            gen_img = self.run_G(gen_z, gen_c, sync=sync)
            gen_logits = self.run_D(gen_img, gen_c, sync=False)

            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())

            loss_Gmain = self.loss.calc_loss(gen_logits, None)
            training_stats.report('Loss/G/loss', loss_Gmain)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss_Gmain.mul(gain).backward()
            
            
    def do_Dmain(self, real_img, real_c, gen_z, gen_c, sync, gain):
        with torch.autograd.profiler.record_function('Dgen_forward'):
            gen_img = self.run_G(gen_z, gen_c, sync=False)
            gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            
            real_img_tmp = real_img.detach().requires_grad_(self.dis_regs is not None)
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/real', real_logits.sign())
            
            loss_Dgen = self.loss.calc_loss(real_logits, gen_logits)
            
        with torch.autograd.profiler.record_function('Dgen_backward'):
            loss_Dgen.mean().mul(gain).backward()
        return real_logits, real_img_tmp
        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Greg  = (phase in ['Greg', 'Gboth']) and self.gen_regs is not None
        do_Dreg  = (phase in ['Dreg', 'Dboth']) and self.dis_regs is not None
        
        if do_Gmain:
            self.do_Gmain(real_img, real_c, gen_z, gen_c, sync=(sync and not do_Greg), gain=gain)
        if do_Greg:
            for i, reg in enumerate(self.gen_regs):
                reg.calc_reg(self, real_img, real_c, gen_z, gen_c, sync=(i==len(self.gen_regs) - 1), gain=gain)
        
        real_logits = None
        real_img_tmp = None
        if do_Dmain:
            real_logits, real_img_tmp = self.do_Dmain(real_img, real_c, gen_z, gen_c, sync=(sync and not do_Dreg), gain=gain)
        
        if do_Dreg:
            if not do_Dmain:
                with torch.autograd.profiler.record_function('Dreg_forward'):
                    real_img_tmp = real_img.detach().requires_grad_(do_Dreg)
                    real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())
            for i, reg in enumerate(self.dis_regs):
                reg.calc_reg(self, real_img, real_c, gen_z, gen_c, real_logits, real_img_tmp, sync=(i==len(self.dis_regs) - 1), gain=gain)


@losses_arch.add_to_registry("base")               
class BasicLoss(LossBase):
    def __init__(self, **args):
        super().__int__(**args)

                
@losses_arch.add_to_registry("sg2")
class SG2Loss(LossBase):  
    def __init__(self, G_mapping = None, G_synthesis = None, style_mixing_prob=0.9, **args):
        assert G_mapping is not None
        assert G_synthesis is not None
        G = dnnlib.EasyDict({'G_mapping': G_mapping, 'G_synthesis' : G_synthesis})
        args.update({'G' : G})
        super().__init__(**args)
        self.style_mixing_prob = style_mixing_prob

        
        
        
    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G.G_mapping, sync):
            ws = self.G.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G.G_synthesis, sync):
            img = self.G.G_synthesis(ws)
        return img
    
    def run_Gws(self, z, c, sync):
        with misc.ddp_sync(self.G.G_mapping, sync):
            ws = self.G.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G.G_synthesis, sync):
            img = self.G.G_synthesis(ws)
        return img, ws


