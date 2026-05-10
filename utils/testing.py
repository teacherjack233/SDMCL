"""
测试工具模块
提供统一的组件性能测试功能，支持多种数据集
"""

import os
import torch
import numpy as np
import logging


def test_components(components, test_loaders, device, args):
    """
    测试所有组件并记录每个任务的准确率
    根据数据集类型自动适配测试逻辑
    """
    # 创建保存目录
    save_dir = "recon_results"
    os.makedirs(save_dir, exist_ok=True)
    
    total_correct_all_tasks = 0
    total_samples_all_tasks = 0
    
    # 禁用梯度计算
    with torch.no_grad():
        for component in components:
            component.vae.eval()  # 将 VAE 模块置于评估模式
            component.classifier.eval()  # 将分类器置于评估模式

        for task_id, test_loader in enumerate(test_loaders):
            total, correct = 0, 0
            
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)

                # 为每个组件计算ELBO值
                elbos = []
                for comp_idx, component in enumerate(components):
                    # 根据组件的输入通道数调整数据
                    if component.input_channels == 1 and data.shape[1] == 3:
                        temp_data = data.mean(dim=1, keepdim=True)
                    elif component.input_channels == 3 and data.shape[1] == 1:
                        temp_data = data.repeat(1, 3, 1, 1)
                    else:
                        temp_data = data
                    
                    # 调整数据形状
                    temp_data = temp_data.reshape([temp_data.size(0), component.input_channels, 32, 32])
                    spike_input = temp_data.unsqueeze(-1).repeat(1, 1, 1, 1, args.n_steps)
                    
                    # 重复输入以增加样本数量
                    spike_input_repeated = spike_input.repeat(1, 1, 1, 1, 1)
                    x_recon, q_z, p_z, sampled_z = component.vae(spike_input_repeated, scheduled=True)
                    
                    # 计算ELBO损失
                    reconstruction_data = temp_data.view([-1, component.input_channels, 32, 32]).repeat(1, 1, 1, 1)
                    elbo = component.vae.loss_function_mmd(
                        reconstruction_data,
                        x_recon, q_z, p_z
                    )
                    avg_elbo = elbo['loss'].mean().item()
                    elbos.append(avg_elbo)

                # 找到最优组件
                best_component_idx = np.argmin(elbos)
                best_component = components[best_component_idx]

                # 使用最佳组件进行分类预测
                predictions = best_component.test_classifier(data)
                batch_matches = (predictions == labels)

                # 计算批次准确率
                batch_size = labels.size(0)
                correct_count = batch_matches.sum().item()
                is_majority_correct = correct_count > (batch_size // 2)

                # 更新统计
                if is_majority_correct:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            logging.info(f"任务 {task_id + 1} 准确率: {accuracy * 100:.2f}%")
            print(f"任务 {task_id + 1} 准确率: {accuracy * 100:.2f}%")
            total_correct_all_tasks += correct
            total_samples_all_tasks += total

    # 计算所有任务的总体准确率
    overall_accuracy = total_correct_all_tasks / total_samples_all_tasks if total_samples_all_tasks > 0 else 0
    logging.info(f"所有任务的总体准确率: {overall_accuracy * 100:.2f}%")
    print(f"所有任务的总体准确率: {overall_accuracy * 100:.2f}%")

    return overall_accuracy