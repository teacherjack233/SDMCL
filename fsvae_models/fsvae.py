
import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F




class FSVAE(nn.Module):
    def __init__(self,in_channels=1,n_steps=8):
        super().__init__()

        self.in_channels = in_channels
        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = n_steps

        self.k = 20

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k,self.n_steps)
        
        self.posterior = PosteriorBernoulliSTBP(self.n_steps,self.k)

        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=self.in_channels,
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer(self.n_steps)

        self.psp = PSP()

    def forward(self, x, scheduled=True):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z
    
    def encode(self, x, scheduled=True):
        x = self.encoder(x) # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3) # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x) # (N,latent_dim,T)
        sampled_z, q_z = self.posterior(latent_x) # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)
        # sampled_z = sampled_z.view([sampled_z.shape[0], self.latent_dim, 1,1,8])
        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def get_check_mmd_loss(self,x,scheduled=True):
        with torch.no_grad():
            x = self.encoder(x)  # (N,C,H,W,T)
            x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
            latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)
            sampled_z, q_z = self.posterior(latent_x)  # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)
            p_z = self.prior(sampled_z, scheduled, self.p)
            q_z_ber = torch.mean(q_z, dim=2)  # (N, latent_dim, T)
            p_z_ber = torch.mean(p_z, dim=2)  # (N, latent_dim, T)

            # kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
            mmd_loss = torch.mean((self.psp(q_z_ber) - self.psp(p_z_ber)) ** 2)
            return  mmd_loss

    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

    def get_sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z


    def loss_function_mse(self, input_img, recons_img):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        return recons_loss


    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': mmd_loss}

    def batch_loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        Per-sample FSVAE loss for memory selection.
        q_z and p_z are (N, latent_dim, k, T).
        """
        recons_loss = F.mse_loss(recons_img, input_img, reduction='none')
        recons_loss = recons_loss.view(recons_loss.size(0), -1).mean(dim=1)

        q_z_ber = torch.mean(q_z, dim=2)
        p_z_ber = torch.mean(p_z, dim=2)
        mmd_loss = torch.mean((self.psp(q_z_ber) - self.psp(p_z_ber)) ** 2, dim=(1, 2))

        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': mmd_loss}

    def latent_mmd(self, source_z, target_z):
        """
        FSVAE-style PSP MMD between two encoded latent spike/probability sets.
        Accepts (N,C,T), (N,C,k,T), or (N,C,1,1,T).
        """
        if source_z.dim() == 4:
            source_z = torch.mean(source_z, dim=2)
        elif source_z.dim() == 5:
            source_z = torch.mean(source_z, dim=(2, 3))

        if target_z.dim() == 4:
            target_z = torch.mean(target_z, dim=2)
        elif target_z.dim() == 5:
            target_z = torch.mean(target_z, dim=(2, 3))

        source_trace = self.psp(source_z.float()).mean(dim=0)
        target_trace = self.psp(target_z.float()).mean(dim=0)
        return torch.mean((source_trace - target_trace) ** 2)

    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))

        loss = recons_loss + 1e-4 * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': kld_loss}
    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
        
