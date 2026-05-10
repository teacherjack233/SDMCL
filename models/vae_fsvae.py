import torch.nn as nn

from fsvae_models.fsvae import FSVAE


class SpikingFSVAE(nn.Module):
    """Thin wrapper around the original FSVAE expert backbone."""

    def __init__(self, in_channels=1, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.fsvae = FSVAE(in_channels=in_channels, n_steps=n_steps)

    def forward(self, x, scheduled=True):
        return self.fsvae(x, scheduled=scheduled)

    def loss_function_mmd(self, x, x_recon, q_z, p_z):
        return self.fsvae.loss_function_mmd(x, x_recon, q_z, p_z)

    def batch_loss_function_mmd(self, x, x_recon, q_z, p_z):
        return self.fsvae.batch_loss_function_mmd(x, x_recon, q_z, p_z)

    def latent_mmd(self, source_z, target_z):
        return self.fsvae.latent_mmd(source_z, target_z)

    def get_check_mmd_loss(self, x):
        return self.fsvae.get_check_mmd_loss(x)

    def encode(self, x):
        return self.fsvae.encode(x)

    def decode(self, z):
        return self.fsvae.decode(z)

    def get_sample(self, num_samples=64, device=None):
        return self.fsvae.get_sample(num_samples)

    def update_p(self, epoch, total_epochs):
        self.fsvae.update_p(epoch, total_epochs)
