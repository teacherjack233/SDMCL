# data/dataloaders.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.memory import PermutedMNIST
import random


def get_transform(dataset_name, img_size=32):
    """根据数据集名称返回相应的预处理变换"""
    if dataset_name.lower() == "mnist":
        # MNIST是单通道，使用单通道归一化
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
        ])
    elif dataset_name.lower() in ["cifar10", "cifar100"]:
        # CIFAR10/CIFAR100是三通道，使用三通道归一化
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 三通道归一化
        ])
    else:
        # 默认使用单通道归一化
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


def get_dataset_class(dataset_name):
    """根据数据集名称返回相应的数据集类"""
    if dataset_name.lower() == "mnist":
        return datasets.MNIST
    elif dataset_name.lower() == "cifar10":
        return datasets.CIFAR10
    elif dataset_name.lower() == "cifar100":
        return datasets.CIFAR100
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def get_task_labels(dataset_name):
    """根据数据集名称返回任务标签划分"""
    if dataset_name.lower() in ["mnist", "cifar10"]:
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    elif dataset_name.lower() == "cifar100":
        # CIFAR100有20个任务，每个任务5个类别
        # 将100个类别分成20组，每组5个类别
        return [(i*5, i*5+1, i*5+2, i*5+3, i*5+4) for i in range(20)]
    else:
        # 默认标签划分
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def get_permuted_mnist_tasks(
    num_tasks=5,
    fraction=0.5,
    batch_size=None,
    train_batch_size=64,
    test_batch_size=10,
    num_workers=0,
    seed=42,
):
    """
    按标签分任务(每个任务两个类别)：(0,1),(2,3),(4,5),(6,7),(8,9)
    并对每个任务做独立像素置换 (permutation)。
    返回:
      task_train_datasets: list of Dataset (可直接用于训练)
      task_test_loaders: list of DataLoader (用于测试)
    """
    assert 0 < fraction <= 1.0
    if batch_size is not None:
        train_batch_size = batch_size
        test_batch_size = batch_size
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 不要覆盖 full dataset，始终以 full 为基准
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 先（可选地）在 full 上做 fraction 下采样，得到 sampled indices（相对于 full）
    if fraction < 1.0:
        sampled_train_idx = np.random.choice(len(full_train), int(len(full_train) * fraction), replace=False)
        sampled_test_idx = np.random.choice(len(full_test), int(len(full_test) * fraction), replace=False)
    else:
        sampled_train_idx = np.arange(len(full_train))
        sampled_test_idx = np.arange(len(full_test))

    # 任务标签划分
    task_labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    assert num_tasks <= len(task_labels)

    task_train_datasets = []
    task_test_loaders = []

    # 将 full.targets 转为 numpy 方便索引（torchvision 里的 targets 通常是 tensor）
    train_targets_np = np.array(full_train.targets)
    test_targets_np = np.array(full_test.targets)

    for task_id in range(num_tasks):
        labels = task_labels[task_id]

        # 从 sampled indices 中筛选出属于当前 task 的那些 full-level indices
        train_idx_in_full = [idx for idx in sampled_train_idx if train_targets_np[idx] in labels]
        test_idx_in_full = [idx for idx in sampled_test_idx if test_targets_np[idx] in labels]

        # 安全检查：确保没有空任务
        if len(train_idx_in_full) == 0 or len(test_idx_in_full) == 0:
            raise RuntimeError(f"Task {task_id} (labels={labels}) has empty split. "
                               "Check fraction or dataset.")

        # 直接基于 full dataset 构建 Subset (indices 对应 full 的下标) —— 这样不会嵌套 Subset 导致索引混淆
        train_subset = Subset(full_train, train_idx_in_full)
        test_subset = Subset(full_test, test_idx_in_full)

        # 每个任务独立的像素置换
        perm = torch.randperm(32 * 32)

        # 包装成 PermutedMNIST，返回的 __getitem__ 会 reshape 回 (1,32,32)
        train_data = PermutedMNIST(train_subset, perm)
        test_data = PermutedMNIST(test_subset, perm)

        task_train_datasets.append(train_data)
        task_test_loaders.append(
            DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
        )

    return task_train_datasets, task_test_loaders


def get_dataset_tasks(
    dataset_name,
    num_tasks=5,
    fraction=0.5,
    batch_size=None,
    train_batch_size=64,
    test_batch_size=10,
    num_workers=0,
    seed=42,
):
    """
    根据数据集名称获取任务数据集
    """
    assert 0 < fraction <= 1.0
    if batch_size is not None:
        train_batch_size = batch_size
        test_batch_size = batch_size
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    # 获取相应的数据集类和变换
    dataset_cls = get_dataset_class(dataset_name)
    transform = get_transform(dataset_name)

    # 加载训练和测试数据集
    if dataset_name.lower() == "mnist":
        full_train = dataset_cls(root='./data', train=True, download=True, transform=transform)
        full_test = dataset_cls(root='./data', train=False, download=True, transform=transform)
    elif dataset_name.lower() in ["cifar10", "cifar100"]:
        full_train = dataset_cls(root='./data', train=True, download=True, transform=transform)
        full_test = dataset_cls(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 先（可选地）在 full 上做 fraction 下采样，得到 sampled indices（相对于 full）
    if fraction < 1.0:
        sampled_train_idx = np.random.choice(len(full_train), int(len(full_train) * fraction), replace=False)
        sampled_test_idx = np.random.choice(len(full_test), int(len(full_test) * fraction), replace=False)
    else:
        sampled_train_idx = np.arange(len(full_train))
        sampled_test_idx = np.arange(len(full_test))

    # 任务标签划分
    task_labels = get_task_labels(dataset_name)
    assert num_tasks <= len(task_labels)

    task_train_datasets = []
    task_test_loaders = []

    # 将 full.targets 转为 numpy 方便索引（torchvision 里的 targets 通常是 tensor）
    train_targets_np = np.array(full_train.targets)
    test_targets_np = np.array(full_test.targets)

    for task_id in range(num_tasks):
        labels = task_labels[task_id]

        # 从 sampled indices 中筛选出属于当前 task 的那些 full-level indices
        if dataset_name.lower() == "cifar100":
            # 对于CIFAR100，标签是一个元组，需要特殊处理
            train_idx_in_full = [idx for idx in sampled_train_idx if train_targets_np[idx] in labels]
            test_idx_in_full = [idx for idx in sampled_test_idx if test_targets_np[idx] in labels]
        else:
            train_idx_in_full = [idx for idx in sampled_train_idx if train_targets_np[idx] in labels]
            test_idx_in_full = [idx for idx in sampled_test_idx if test_targets_np[idx] in labels]

        # 安全检查：确保没有空任务
        if len(train_idx_in_full) == 0 or len(test_idx_in_full) == 0:
            raise RuntimeError(f"Task {task_id} (labels={labels}) has empty split. "
                               "Check fraction or dataset.")

        # 直接基于 full dataset 构建 Subset (indices 对应 full 的下标) —— 这样不会嵌套 Subset 导致索引混淆
        train_subset = Subset(full_train, train_idx_in_full)
        test_subset = Subset(full_test, test_idx_in_full)

        task_train_datasets.append(train_subset)
        task_test_loaders.append(
            DataLoader(test_subset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
        )

    return task_train_datasets, task_test_loaders


def get_task_loader(task_data, batch_size):
    """
    获取特定任务的数据加载器
    """
    return DataLoader(task_data, batch_size=batch_size, shuffle=True, drop_last=True)


def get_test_loader(dataset_name, batch_size=64, task_idx=0, img_size=32):
    """
    获取特定任务的测试集数据加载器
    """
    transform = get_transform(dataset_name, img_size)

    if dataset_name.lower() == "cifar100":
        # CIFAR100有20个任务，每个任务包含5个类别
        task_labels = [(i*5, i*5+1, i*5+2, i*5+3, i*5+4) for i in range(20)]
        labels_to_include = list(task_labels[task_idx])
    else:
        # 其他数据集使用原来的标签划分
        task_labels = [
            [0, 1],  # 任务1: 类别0和1
            [2, 3],  # 任务2: 类别2和3
            [4, 5],  # 任务3: 类别4和5
            [6, 7],  # 任务4: 类别6和7
            [8, 9]   # 任务5: 类别8和9
        ]
        labels_to_include = task_labels[task_idx]

    dataset_cls = get_dataset_class(dataset_name)
    dataset = dataset_cls(root='./data', train=False, download=True, transform=transform)

    # 根据数据集类型获取targets
    if dataset_name.lower() in ["cifar10", "cifar100"]:
        targets = dataset.targets
        mask = torch.tensor([label in labels_to_include for label in targets])
        dataset.data = dataset.data[mask.numpy()]
        dataset.targets = torch.tensor(targets)[mask]
    else:  # MNIST
        targets = dataset.targets
        mask = torch.tensor([label in labels_to_include for label in targets])
        dataset.data = dataset.data[mask]
        dataset.targets = targets[mask]

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
