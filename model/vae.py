import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# hyper parameters
latent_dim = 8


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 2 * latent_dim))

    def forward(self, x, cond):
        x_cond = torch.cat([x, cond], dim=1)
        h = self.net(x_cond)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim + 2, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, 4))

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=1)
        return self.net(z_cond)


class CVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar
