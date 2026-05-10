# utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os


def calculate_mmd(z1, z2, sigma=1.0):
    """
    计算两个潜在空间 z1 和 z2 的 MMD 距离
    :param z1: 第一个分布的样本 (N, D)
    :param z2: 第二个分布的样本 (N, D)
    :param sigma: RBF 核的参数
    :return: MMD 距离
    """
    # 样本大小
    N = z1.size(0)

    # 计算核矩阵
    K_xx = rbf_kernel(z1, z1, sigma)  # z1 与 z1 的核矩阵
    K_yy = rbf_kernel(z2, z2, sigma)  # z2 与 z2 的核矩阵
    K_xy = rbf_kernel(z1, z2, sigma)  # z1 与 z2 的核矩阵

    # 计算经验 MMD（保留对角线）
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

    return mmd.item()


def rbf_kernel(x, y, sigma=1.0):
    """
    RBF核函数，计算x和y之间的相似度
    :param x: 第一个样本组 (N, D)
    :param y: 第二个样本组 (M, D)
    :param sigma: 核函数的标准差参数
    :return: RBF核矩阵 (N, M)
    """
    beta = 1 / (2 * sigma ** 2)
    dist = torch.cdist(x, y, p=2)  # 计算 x 和 y 之间的欧氏距离
    return torch.exp(-beta * dist ** 2)


def calculate_mse(z1, z2):
    """
    计算两个潜在空间 z1 和 z2 之间的 MSE 距离
    :param z1: 第一个组件生成的潜在变量
    :param z2: 第二个组件生成的潜在变量
    :return: MSE 距离
    """
    mse_loss = F.mse_loss(z1, z2, reduction='sum')
    return mse_loss.item() / 100


def calculate_cosine_similarity(z1, z2):
    """
    计算两个潜在空间 z1 和 z2 之间的余弦相似度
    :param z1: 第一个组件生成的潜在变量
    :param z2: 第二个组件生成的潜在变量
    :return: 余弦相似度
    """
    z1_normalized = F.normalize(z1, p=2, dim=1)  # 对z1进行归一化
    z2_normalized = F.normalize(z2, p=2, dim=1)  # 对z2进行归一化

    # 计算余弦相似度
    cosine_similarity = torch.mm(z1_normalized, z2_normalized.T)
    return cosine_similarity.mean().item()


def plot_mse_similarity_matrix(components, time_dir, filename="mse_similarity_matrix.png"):
    """
    生成 MSE 距离矩阵并保存为 PNG 文件
    :param components: 组件列表
    :param time_dir: 保存目录
    :param filename: 保存的 PNG 文件名
    """
    num_components = len(components)
    mse_matrix = np.zeros((num_components, num_components))
    z_samples = []

    # 对每个组件，使用 get_sample 函数生成样本并通过编码器重新编码获得潜在空间表示
    for component in components:
        component.vae.eval()  # 切换 VAE 到评估模式，确保不会发生梯度计算等
        with torch.no_grad():
            # 直接调用 get_sample 生成样本
            generated_samples, _ = component.vae.get_sample(num_samples=64, device=component.device)
            
            # 根据组件的输入通道数调整数据
            if component.input_channels == 1 and generated_samples.shape[1] == 3:
                # 3通道转单通道
                generated_samples = generated_samples.mean(dim=1, keepdim=True)
            elif component.input_channels == 3 and generated_samples.shape[1] == 1:
                # 单通道转3通道
                generated_samples = generated_samples.repeat(1, 3, 1, 1)
            
            # 确保输入数据格式正确
            # 如果是 (N, C, H, W) 需要扩展为 (N, C, H, W, T)
            if len(generated_samples.shape) == 4:  # (N, C, H, W)
                # 扩展为5维 (N, C, H, W, T)
                generated_samples = generated_samples.unsqueeze(-1).repeat(1, 1, 1, 1, component.vae.n_steps)
            
            # 通过编码器获得潜在表示
            encoded_result = component.vae.encode(generated_samples)
            mu = encoded_result[0]  # 获取第一个返回值 (sampled_z_q)
            
            # 根据VAE的输出形状调整处理方式
            if len(mu.shape) > 2:  # 如果mu仍然有多余维度
                mu_flat = mu.view(mu.size(0), -1)  # 只展平除批次外的维度
            else:
                mu_flat = mu
            
            z_samples.append(mu_flat)  # 存储潜在表示

    # 计算每对组件之间的 MSE 距离
    for i in range(num_components):
        for j in range(i + 1, num_components):
            mse = calculate_mse(z_samples[i], z_samples[j])
            mse_matrix[i, j] = mse
            mse_matrix[j, i] = mse  # 对称矩阵

    # 可视化 MSE 距离矩阵
    plt.figure(figsize=(6, 6))
    plt.imshow(mse_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='MSE Distance')
    plt.title('MSE Distance Matrix')
    plt.xlabel('Component Index')
    plt.ylabel('Component Index')

    # 显示矩阵中的数值
    for i in range(num_components):
        for j in range(num_components):
            plt.text(j, i, f'{mse_matrix[i, j]:.2f}', ha='center', va='center', color='black')

    # 保存为 PNG 文件
    filepath = os.path.join(time_dir, filename)
    plt.savefig(filepath)
    plt.close()


def save_component_samples_as_png(components, num_samples=64, device="cuda", time_dir=None):
    """保存组件生成的样本"""
    # 确保 `time_dir` 是在配置日志时确定的，而不是在每次调用时重新生成
    if time_dir is None:
        raise ValueError("time_dir is not defined! It should be the same directory where logs are saved.")

    # 确保使用GPU进行采样
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 创建不同的目录用于保存VAE和GAN的样本
    vae_output_dir = os.path.join(time_dir, "vae_samples")
    os.makedirs(vae_output_dir, exist_ok=True)

    for idx, component in enumerate(components):
        component.vae.eval()  # 确保VAE在评估模式下
        with torch.no_grad():
            # 生成VAE样本
            vae_samples = component.vae.get_sample(num_samples)[0].to(device)

        # 将VAE生成的样本保存为PNG文件
        vae_filename = os.path.join(vae_output_dir, f"vae_samples_component_{idx + 1}.png")
        save_image(vae_samples, vae_filename, nrow=8, normalize=True)
        print(f"已保存组件 {idx + 1} 的VAE样本到 {vae_filename}")
