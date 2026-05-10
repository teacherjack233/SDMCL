# training/expansion.py
import torch
import torch.nn.functional as F
import numpy as np
# 修复相对导入问题
from models.component import Component
from fsvae_models.esvae import MMD_loss
import logging


def compute_pairwise_distance(comp1, comp2, num_samples=64, n_steps=8):
    """计算两个组件之间的差异"""
    print(f"\n🔍 正在计算组件 {comp1} 和 {comp2} 的差异...")
    # 交叉编码流程
    z1_self = cross_encode(comp1, comp1, num_samples, n_steps)
    z1_cross = cross_encode(comp1, comp2, num_samples, n_steps)
    z2_self = cross_encode(comp2, comp2, num_samples, n_steps)
    z2_cross = cross_encode(comp2, comp1, num_samples, n_steps)

    # 计算双向MSE
    mse_loss = torch.nn.MSELoss()
    mse1 = mse_loss(z1_self, z1_cross).item()
    mse2 = mse_loss(z2_self, z2_cross).item()

    print(f"  组件对 MSE: {mse1:.4f} (正向) + {mse2:.4f} (反向)")
    return (mse1 + mse2) / 2


def cross_encode(gen_component, encode_component, num_samples, n_steps):
    """生成并交叉编码样本"""
    with torch.no_grad():
        # 生成源组件样本
        gen_samples, _ = gen_component.vae.get_sample(num_samples)  # 移除了device参数

        # 检查生成样本的通道数是否与编码器期望的通道数一致
        # 我们知道CIFAR100组件的input_channels是3，MNIST是1
        # 从组件对象获取期望的通道数
        expected_channels = encode_component.input_channels
        actual_channels = gen_samples.shape[1]
        
        if actual_channels != expected_channels:
            if expected_channels == 3 and actual_channels == 1:
                # 单通道转三通道
                gen_samples = gen_samples.repeat(1, 3, 1, 1)
            elif expected_channels == 1 and actual_channels == 3:
                # 三通道转单通道
                gen_samples = gen_samples.mean(dim=1, keepdim=True)
        
        # 添加时间维度 - 确保数据格式正确
        if len(gen_samples.shape) == 4:  # [N, C, H, W]
            spike_input = gen_samples.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # [N, C, H, W, T]
        else:  # 已经是5维
            spike_input = gen_samples

        # 使用目标组件编码
        z_spikes, _, _ = encode_component.vae.encode(spike_input)

        return torch.mean(z_spikes.float(), dim=[0, 2])  # [B, N]


def get_sample_z(gen_component, encode_component, num_samples, n_steps):
    """生成并交叉编码样本"""
    with torch.no_grad():
        # 生成源组件样本
        gen_samples, _ = gen_component.vae.get_sample(num_samples)  # 移除了device参数
        
        # 检查生成样本的通道数是否与编码器期望的通道数一致
        # 从组件对象获取期望的通道数
        expected_channels = encode_component.input_channels
        actual_channels = gen_samples.shape[1]
        
        if actual_channels != expected_channels:
            if expected_channels == 3 and actual_channels == 1:
                # 单通道转三通道
                gen_samples = gen_samples.repeat(1, 3, 1, 1)
            elif expected_channels == 1 and actual_channels == 3:
                # 三通道转单通道
                gen_samples = gen_samples.mean(dim=1, keepdim=True)
        
        # 添加时间维度 - 确保数据格式正确
        if len(gen_samples.shape) == 4:  # [N, C, H, W]
            spike_input = gen_samples.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # [N, C, H, W, T]
        else:  # 已经是5维
            spike_input = gen_samples
            
        # 使用目标组件编码
        z_spikes, r_q, _ = encode_component.vae.encode(spike_input)
        return r_q


def get_sample_z_from_data(encode_component, data, n_steps):
    """从给定数据中编码样本"""
    with torch.no_grad():
        # 确保数据格式正确 - 添加时间维度
        if len(data.shape) == 4:  # [N, C, H, W]
            # 检查通道数是否与VAE期望的一致
            expected_channels = encode_component.input_channels  # 从组件获取通道数
            actual_channels = data.shape[1]
            
            if actual_channels != expected_channels:
                if expected_channels == 3 and actual_channels == 1:
                    # 单通道转三通道
                    data = data.repeat(1, 3, 1, 1)
                elif expected_channels == 1 and actual_channels == 3:
                    # 三通道转单通道
                    data = data.mean(dim=1, keepdim=True)
            
            spike_input = data.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # [N, C, H, W, T]
        else:  # 已经是5维
            spike_input = data
        # 使用目标组件编码
        z_spikes, r_q, _ = encode_component.vae.encode(spike_input)
        return r_q


def check_expansion_mmd(experts, data, threshold, n_steps=8):
    """
    计算前n-1个专家与data之间的MMD损失，返回是否扩展和最小MMD值

    参数:
        experts: 专家列表，每个专家包含VAE模型
        data: 输入数据张量，形状为(batch_size, ...)
        threshold: MMD阈值，决定是否扩展新专家
        n_steps: 时间步数

    返回:
        should_expand: 布尔值，表示是否应该扩展新专家
        min_mmd: 最小的MMD损失值
    """
    n = len(experts)
    if n <= 1:
        # 如果专家数量不足，默认需要扩展
        return True, torch.tensor(float('inf'), device=data.device)

    mmd_loss_fn = MMD_loss(kernel_type='rbf')  # 使用RBF核计算MMD
    min_mmd = float('inf')
    num_samples = data.size(0)  # 使用data的batch_size作为采样数量

    # 只计算前n-1个专家
    for i in range(n - 1):
        expert = experts[i]

        # 获取专家生成的样本编码
        z_expert = get_sample_z(
            gen_component=expert,
            encode_component=expert,
            num_samples=num_samples,
            n_steps=n_steps
        )

        # 获取data在专家编码器中的表示
        z_data = get_sample_z_from_data(
            encode_component=expert,
            data=data,
            n_steps=n_steps
        )

        # 展平特征维度以匹配MMD计算要求
        z_expert_flat = z_expert.view(z_expert.size(0), -1)
        z_data_flat = z_data.view(z_data.size(0), -1)

        # 计算MMD损失
        current_mmd = mmd_loss_fn(z_expert_flat, z_data_flat)

        # 更新最小MMD值
        if current_mmd.item() < min_mmd:
            min_mmd = current_mmd.item()

    min_mmd_tensor = torch.tensor(min_mmd, device=data.device)

    # 判断是否扩展：如果所有现有专家的MMD都大于阈值，则需要扩展新专家
    should_expand = min_mmd > threshold

    return should_expand, min_mmd_tensor


def check_expansion_fire(components, threshold=0.02, num_samples=64, n_steps=8):
    logging.info(f"=== Torch Expansion Check (Threshold: {threshold}, Samples: {num_samples}) ===")
    print(f"\nThreshold: {threshold}, Samples/component: {num_samples}")
    logging.info(f"\nThreshold: {threshold}, Samples/component: {num_samples}")

    if len(components) < 2:
        logging.info("Immediate expansion: component count < 2")
        print("Automatic expansion: Initial component count < 2")
        return True, 0

    # 获取最新组件和旧组件列表
    new_component = components[-1]
    prev_components = components[:-1]

    # 计算所有旧组件与最新组件的MSE
    mse_values = []
    for idx, old_comp in enumerate(prev_components):
        print(f"\nComparing Component {idx} (old) vs {len(components)-1} (new)")
        logging.info(f"\nComparing Component {idx} (old) vs {len(components)-1} (new)")
        mse = compute_pairwise_distance(old_comp, new_component, num_samples, n_steps)
        mse_values.append(mse)
        print(f"  Pair MSE[{idx}-{len(components)-1}]: {mse:.4f}")
        logging.info(f"  Pair MSE[{idx}-{len(components)-1}]: {mse:.4f}")

    # 取最小值与阈值比较
    min_mse = min(mse_values)
    print(f"\nMinimum pairwise MSE: {min_mse:.4f} (Threshold: {threshold})")
    logging.info(f"\nMinimum pairwise MSE: {min_mse:.4f} (Threshold: {threshold})")

    if min_mse > threshold:
        print("✅ Expansion triggered: All existing components are distinct")
        logging.info("✅ Expansion triggered: All existing components are distinct")
        return True, min_mse
    else:
        print("⏸️ No expansion: Found similar existing component")
        logging.info("⏸️ No expansion: Found similar existing component")
        return False, min_mse