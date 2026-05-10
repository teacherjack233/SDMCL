import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F


class Batch_MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(Batch_MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # 使用广播机制高效计算L2距离
        total_expand1 = total.unsqueeze(1)  # [n, 1, d]
        total_expand2 = total.unsqueeze(0)  # [1, n, d]
        L2_distance = torch.sum((total_expand1 - total_expand2) ** 2, dim=2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 向量化计算所有核函数
        kernel_val = torch.zeros_like(L2_distance)
        for bandwidth_temp in bandwidth_list:
            kernel_val += torch.exp(-L2_distance / bandwidth_temp)

        return kernel_val

    def linear_mmd2(self, source, target):
        # 高效向量化计算线性MMD
        delta = source - target
        return torch.sum(delta * delta, dim=1)

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = source.size(0)
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

            # 高效提取各部分核矩阵
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]

            # 向量化计算每个样本的MMD贡献
            XX_i = torch.mean(XX, dim=1)
            YY_i = torch.mean(YY, dim=1)
            XY_i = torch.mean(XY, dim=1)

            return XX_i + YY_i - 2 * XY_i

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class SampledSpikeAct(torch.autograd.Function):
    """
        Implementation of the spiking activation function with an approximation of gradient.
    """

    @staticmethod
    def forward(ctx, input):
        random_sign = torch.rand_like(input, dtype=input.dtype).to(input.device)
        ctx.save_for_backward(input, random_sign)
        # if input = u > Vth then output = 1
        output = torch.gt(input, random_sign)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, random_sign = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input - random_sign) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu


class ESVAE(nn.Module):
    def __init__(self, in_channels,n_step):
        super().__init__()

        self.in_channels = in_channels
        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = n_step

        self.device = 0
        self.distance_lambda = 0.001
        self.mmd_type = 'rbf'

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
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)

        self.posterior = PosteriorBernoulliSTBP(self.k)

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
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
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

        self.membrane_output_layer = MembraneOutputLayer(n_steps=n_step)

        self.psp = PSP()

        self.sample_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )

        self.mmd_loss = MMD_loss(kernel_type=self.mmd_type)
        self.batch_mmd_loss = Batch_MMD_loss(kernel_type=self.mmd_type)

    def loss_function_mse(self, input_img, recons_img):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        return recons_loss

    def forward(self, x, scheduled=False):
        sampled_z_q, r_q, r_p = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z_q)
        # print(x_recon.shape)
        # print(r_q.shape)
        return x_recon, r_q, r_p, sampled_z_q

    def encode(self, x, scheduled=False, return_firing_rate=False):
        x = self.encoder(x)  # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)

        sampled_z_q, r_q, r_p = self.gaussian_sample(latent_x, latent_x.shape[0])
        return sampled_z_q, r_q, r_p

    def get_check_mmd_loss(self, x, scheduled=False):
        with torch.no_grad():
            _, r_q, r_p = self.encode(x, scheduled=scheduled)
            return self.mmd_loss(r_q, r_p)

    def decode(self, z):
        result = self.decoder_input(z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)
        result = self.decoder(result)  # (N,C,H,W,T)
        result = self.final_layer(result)  # (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))
        return out

    def sample(self, batch_size=64):
        sampled_z_p, _, _ = self.gaussian_sample(batch_size=batch_size)
        sampled_x = self.decode(sampled_z_p)
        return sampled_x, sampled_z_p

    def get_sample(self, batch_size=64):
        sampled_z_p, _, _ = self.gaussian_sample(batch_size=batch_size)
        sampled_x = self.decode(sampled_z_p)
        return sampled_x, sampled_z_p
    def gaussian_sample(self, latent_x=None, batch_size=None, mu=None, var=None):
        if latent_x is not None:
            sampled_z_n = torch.randn(
                (batch_size, self.latent_dim),
                device=latent_x.device,
            )  # (N, latent_dim)
            r_p = self.sample_layer(sampled_z_n)

            r_q = latent_x.mean(-1, keepdim=True).repeat((1, 1, self.n_steps))
            sampled_z_q = SampledSpikeAct.apply(r_q)

            r_q = latent_x.mean(-1)   # (N, latent_dim)

            return sampled_z_q, r_q, r_p
        else:
            device = next(self.parameters()).device
            sampled_z_n = torch.randn((batch_size, self.latent_dim), device=device)
            # if mu is None and var is None:
            #     mu = self.mu
            #     var = self.var
            # var = var * torch.ones_like(sampled_p).to(self.device)  # (N, latent_dim)
            # mu = mu * torch.ones_like(sampled_p).to(self.device)  # (N, latent_dim)
            # sampled_p = mu + sampled_z_n * var
            r_p = self.sample_layer(sampled_z_n)
            r_p = r_p.unsqueeze(dim=-1).repeat((1, 1, self.n_steps))
            sampled_z_q = SampledSpikeAct.apply(r_p)
            return sampled_z_q, None, None

    def batch_loss_function_mmd(self, input_img, recons_img, r_q, r_p):
        """
        安全的批量损失计算（避免显存溢出）
        r_q is q(z|x): (N,latent_dim)
        r_p is p(z): (N,latent_dim)
        """

        # 计算重构损失（每个样本）
        recons_loss = F.mse_loss(recons_img, input_img, reduction='none')
        recons_loss = recons_loss.view(recons_loss.size(0), -1).mean(dim=1)
        # 初始化MMD损失
        mmd_loss = self.batch_mmd_loss(r_q, r_p)

        # 组合总损失
        loss = recons_loss + self.distance_lambda * mmd_loss

        return {
            'loss': loss,  # 每个样本的损失值 [batch_size]
            'Reconstruction_Loss': recons_loss,
            'Distance_Loss': mmd_loss
        }
    def loss_function_mmd(self, input_img, recons_img, r_q, r_p):
        """
        r_q is q(z|x): (N,latent_dim)
        r_p is p(z): (N,latent_dim)
        """
        # print(recons_img.shape)
        # print(input_img.shape)
        recons_loss = F.mse_loss(recons_img, input_img)
        # print(r_p.shape, r_p.max(), r_p.min(), r_p.mean())
        # print(r_q.shape, r_q.max(), r_q.min(), r_q.mean())
        mmd_loss = self.mmd_loss(r_q, r_p)

        loss = recons_loss + self.distance_lambda * mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': mmd_loss}

    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4, 4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


