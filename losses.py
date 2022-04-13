import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

losses = utils.ClassRegistry()


@losses.add_to_registry("bcew")
class BCEWithLogits(nn.BCEWithLogitsLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            loss = super().forward(pred_real, torch.ones_like(pred_real))
            return loss


@losses.add_to_registry("hinge")        
class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            return loss_real + loss_fake
        else:
            loss = -pred_real.mean()
            return loss


@losses.add_to_registry("wasserstein")
class Wasserstein(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = -pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = -pred_real.mean()
            return loss


@losses.add_to_registry("softplus")
class Softplus(nn.Module):
    def forward(self, pred_real, pred_fake=None): # поменять местами real и fake
        if pred_fake is not None:
            loss_real = F.softplus(-pred_real).mean()
            loss_fake = F.softplus(pred_fake).mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = F.softplus(-pred_real).mean()
            return loss
