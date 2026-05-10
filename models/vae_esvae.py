# models/vae_esvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsvae_models.esvae import ESVAE


class SpikingESVAE(nn.Module):
    def __init__(self, in_channels=1, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        self.esvae = ESVAE(in_channels=in_channels, n_step=n_steps)
    
    def forward(self, x, scheduled=True):
        return self.esvae(x, scheduled=scheduled)
    
    def loss_function_mmd(self, x, x_recon, q_z, p_z):
        return self.esvae.loss_function_mmd(x, x_recon, q_z, p_z)
    
    def batch_loss_function_mmd(self, x, x_recon, q_z, p_z):
        return self.esvae.batch_loss_function_mmd(x, x_recon, q_z, p_z)
    
    def get_check_mmd_loss(self, x):
        return self.esvae.get_check_mmd_loss(x)
    
    def encode(self, x):
        return self.esvae.encode(x)
    
    def reparameterize(self, mu, logvar):
        return self.esvae.reparameterize(mu, logvar)
    
    def decode(self, z):
        return self.esvae.decode(z)
    
    def sample(self, num_samples=1, device='cpu'):
        return self.esvae.sample(num_samples, device)
    
    def get_sample(self, num_samples=64, device=None):
        # 修复方法，只传递num_samples参数，因为原始方法不接受device参数
        return self.esvae.get_sample(num_samples)
    
    def update_p(self, epoch, total_epochs):
        self.esvae.update_p(epoch, total_epochs)