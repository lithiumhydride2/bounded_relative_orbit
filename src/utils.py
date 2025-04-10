import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vae import latent_dim


def loss_fn(x_recon, x_true, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x_true)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def gengrate_sample(model, cond, num_samples=100, sample_std=1.0):
    """
    Generate samples from the CVAE model.
    :param model: Trained CVAE model.
    :param cond: Condition input for the decoder.
    :param num_samples: Number of samples to generate.
    :param sample_std: Standard deviation for the latent space sampling.
    :return: Generated samples.
    """
    model.eval()
    cond = torch.tensor(cond, dtype=torch.float32).unsqueeze(0)
    cond = cond.repeat(num_samples, 1)

    z = torch.randn(num_samples, latent_dim) * sample_std
    with torch.no_grad():
        samples = model.decoder(z, cond)
    return samples.numpy()
