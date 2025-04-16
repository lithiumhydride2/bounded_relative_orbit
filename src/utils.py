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


def gengrate_sample_mle(f_model,
                        target_y,
                        num_candidates=10000,
                        num_sample=20):
    f_model.eval()
    with torch.no_grad():

        x_candidates = torch.rand(num_candidates, 4)  # 假设 [0,1] 区间

        y_preds = f_model(x_candidates)

        target_tensor = torch.tensor(target_y,
                                     dtype=torch.float32).unsqueeze(0)
        errors = torch.norm(y_preds - target_tensor, dim=1)

        top_indices = torch.topk(-errors, k=num_sample).indices
        return x_candidates[top_indices]
