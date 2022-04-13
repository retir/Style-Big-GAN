import utils
from stylegan2ada.torch_utils.ops import conv2d_gradfix

generator_regs = utils.ClassRegistry()
discriminator_regs = utils.ClassRegistry()


@generator_regs.add_to_registry("ppl")
class PPLreg:
    def __init__(self, device, pl_batch_shrink, pl_decay, pl_weight):
        self.device = device
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        
    def calc_reg(self, model, real_img, real_c, gen_z, gen_c, sync, gain):
        if hasattr(model.G, 'G_mapping'):
            with torch.autograd.profiler.record_function('Gpl_forward'):
                if self.pl_weight != 0:
                    batch_size = gen_z.shape[0] // self.pl_batch_shrink
                    gen_img, gen_ws = model.run_Gws(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                    pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                    with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                    pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                    self.pl_mean.copy_(pl_mean.detach())
                    pl_penalty = (pl_lengths - pl_mean).square()
                    training_stats.report('Loss/pl_penalty', pl_penalty)
                    loss_Gpl = pl_penalty * self.pl_weight
                    training_stats.report('Loss/G/reg', loss_Gpl)
                with torch.autograd.profiler.record_function('Gpl_backward'):
                    (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
                

@discriminator_regs.add_to_registry("r1")
class R1reg:
    def __init__(self, device, r1_gamma):
        self.device = device
        self.r1_gamma = r1_gamma
        
        def calc_reg(self, model, real_img, real_c, gen_z, gen_c, real_logits, real_img_tmp, sync, gain):
            if self.r1_gamma != 0:
                with torch.autograd.profiler.record_function('Dr1_forward'):
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/r1reg', loss_Dr1)

                with torch.autograd.profiler.record_function('Dr1_backward'):
                    (real_logits * 0 + loss_Dr1).mean().mul(gain).backward()