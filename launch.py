import os

import torch
import torch.optim as optim
import wandb
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange
from pytorch_gan_metrics import get_inception_score_and_fid

import source.losses as losses
from source.utils import generate_imgs, infiniteloop, set_seed
from source.gens import net_G_models
from source.discs import net_D_models
from omegaconf import OmegaConf


loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}

device = torch.device('cuda:0')


def generate(config):
    model_config = config.model_params[config.gen.model]
    assert config.gen.pretrain is not None, "set model weight by --pretrain [model]"

    net_G = net_G_models[model_config.arch](model_config.z_dim).to(device)
    net_G.load_state_dict(torch.load(config.gen.pretrain)['net_G'])
    net_G.eval()

    counter = 0
    os.makedirs(config.gen.output)
    with torch.no_grad():
        for start in trange(
                0, model_config.num_images, model_config.batch_size, dynamic_ncols=True):
            batch_size = min(model_config.batch_size, model_config.num_images - start)
            z = torch.randn(batch_size, model_config.z_dim).to(device)
            x = net_G(z).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(
                    image, os.path.join(config.gen.output, '%d.png' % counter))
                counter += 1


def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp


def train(config):
    if config.exp.pretrain:
        model_path = os.path.join(config.exp.pretrain_path, 'model.pt')
        checkpoint = torch.load(model_path)
        config_path = os.path.join(config.exp.pretrain_path, 'config.yaml')
        config = OmegaConf.load(config_path)
        config.exp.pretrain = True

    wandb.init(project=config.exp.project, entity="retir", name=config.exp.name)
    wandb.config = config
    model_config = config.model_params[config.gen.model]
    if model_config.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if model_config.dataset == 'stl10':
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=model_config.batch_size, shuffle=True, num_workers=4,
        drop_last=True)

    net_G = net_G_models[model_config.arch](model_config.z_dim).to(device)
    net_D = net_D_models[model_config.arch]().to(device)
    loss_fn = loss_fns[model_config.loss]()

    optim_G = optim.Adam(net_G.parameters(), lr=model_config.lr_G, betas=[model_config.beta_0, model_config.beta_1])
    optim_D = optim.Adam(net_D.parameters(), lr=model_config.lr_D, betas=[model_config.beta_0, model_config.beta_1])
    sched_G = optim.lr_scheduler.LambdaLR(
        optim_G, lambda step: 1 - step / model_config.total_steps)
    sched_D = optim.lr_scheduler.LambdaLR(
        optim_D, lambda step: 1 - step / model_config.total_steps)

    if config.exp.pretrain:
        net_G.load_state_dict(checkpoint['net_G'])
        net_D.load_state_dict(checkpoint['net_D'])

        optim_G.load_state_dict(checkpoint['optim_G'])
        optim_D.load_state_dict(checkpoint['optim_D'])

        sched_G.load_state_dict(checkpoint['sched_G'])
        sched_D.load_state_dict(checkpoint['sched_D'])
    try:
        os.makedirs(os.path.join(config.log.logdir, config.exp.name))
    except FileExistsError:
      pass

    try:
        os.makedirs(os.path.join(config.log.logdir + f'/{config.exp.name}', 'sample'))
    except FileExistsError:
      pass

    sample_z = torch.randn(config.log.sample_size, model_config.z_dim).to(device)

    if not config.exp.pretrain:
        with open(os.path.join(config.log.logdir + f'/{config.exp.name}', "config.yaml"), 'w') as f:
            OmegaConf.save(config=config, f=f.name)

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:config.log.sample_size]) + 1) / 2

    looper = infiniteloop(dataloader)
    start_step = 1
    if config.exp.pretrain:
        start_step = checkpoint['last_step'] + 1
    with trange(start_step, model_config.total_steps + 1, desc='Training') as pbar:
        for step in pbar:
            # Discriminator
            for _ in range(model_config.n_dis):
                with torch.no_grad():
                    z = torch.randn(model_config.batch_size, model_config.z_dim).to(device)
                    fake = net_G(z).detach()
                real = next(looper).to(device)
                net_D_real = net_D(real)
                net_D_fake = net_D(fake)
                loss = loss_fn(net_D_real, net_D_fake)
                if config.gen.model != 'wgangp':
                    wandb.log({"loss_dis": loss})
                if config.gen.model == 'wgangp':
                    loss_gp = cacl_gradient_penalty(net_D, real, fake)  # wgangp
                    loss_all = loss + model_config.alpha * loss_gp
                    wandb.log({"loss_dis": loss_all})

                optim_D.zero_grad()
                loss.backward()
                optim_D.step()

                if config.gen.model == 'wgan':
                    for p in net_D.parameters():          # wgan
                        p.data.clamp_(-model_config.c, model_config.c)

                if model_config.loss == 'was':
                    loss = -loss
                pbar.set_postfix(loss='%.4f' % loss)
            

            # Generator
            if config.gen.model == 'wgangp':
                for p in net_D.parameters():  # wgangp
                    p.requires_grad_(False)
            z = torch.randn(model_config.batch_size * 2, model_config.z_dim).to(device)
            loss = loss_fn(net_D(net_G(z)))
            wandb.log({"loss_gen": loss})

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()
            if config.gen.model == 'wgangp':
                for p in net_D.parameters():  # wgangp
                    p.requires_grad_(True)

            sched_G.step()
            sched_D.step()

            if step == 1 or step % config.log.sample_step == 0:
                fake = net_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                save_image(grid, os.path.join(
                    config.log.logdir + f'/{config.exp.name}', 'sample', '%d.png' % step))

            if step == 1 or step % config.log.eval_step == 0:
                torch.save({
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                    'last_step' : step,
                }, os.path.join(config.log.logdir + f'/{config.exp.name}', 'model.pt'))
                if config.log.record:
                    imgs = generate_imgs(
                        net_G, device, model_config.z_dim,
                        config.gen.num_images, model_config.batch_size)
                    IS, FID = get_inception_score_and_fid(
                        imgs, config.log.fid_cache, verbose=True)
                    wandb.log({"IS": IS[0]})
                    wandb.log({"FID": FID})
                    with open(os.path.join(config.log.logdir + f'/{config.exp.name}', "logs.txt"), 'a') as f:
                        format_str = f'Step: {step}, FID: {FID:.4f}, IS: {IS[0]:.4f} +- {IS[1]:.4f}\n'
                        f.write(format_str)
                    pbar.write(
                        "%s/%s Inception Score: %.3f(%.5f), "
                        "FID: %6.3f" % (
                            step, model_config.total_steps, IS[0], IS[1], FID))



def launcher(config):
    model_config = config.model_params[config.gen.model]
    set_seed(model_config.seed)
    if config.gen.generate:
        generate()
    else:
        train(config)

