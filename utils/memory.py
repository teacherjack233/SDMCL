# utils/memory.py
import random
import torch
import numpy as np
from torch.utils.data import Subset


class MemoryBuffer:
    """
    内存缓冲区，用于存储样本及其标签
    """

    def __init__(self, size=1000, input_channels=1):
        self.size = size
        self.buffer = []
        self.input_channels = input_channels  # 记录输入通道数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_samples(self, samples, labels):
        # 确保样本具有正确的通道数
        processed_samples = []
        for sample in samples:
            sample_tensor = torch.tensor(sample, dtype=torch.float32)
            if len(sample_tensor.shape) == 3:  # [C, H, W]
                if sample_tensor.shape[0] != self.input_channels:
                    if self.input_channels == 3 and sample_tensor.shape[0] == 1:
                        # 单通道转三通道
                        sample_tensor = sample_tensor.repeat(3, 1, 1)
                    elif self.input_channels == 1 and sample_tensor.shape[0] == 3:
                        # 三通道转单通道
                        sample_tensor = sample_tensor.mean(dim=0, keepdim=True)
            elif len(sample_tensor.shape) == 2:  # [H, W] - 单通道
                sample_tensor = sample_tensor.unsqueeze(0)  # 添加通道维度
                if self.input_channels == 3:
                    sample_tensor = sample_tensor.repeat(3, 1, 1)  # 单通道转三通道
            
            processed_samples.append(sample_tensor.numpy())
        
        combined = list(zip(processed_samples, labels))
        if len(self.buffer) + len(combined) > self.size:
            return False
        self.buffer.extend(combined)
        return True

    def get_samples(self, shuffle=True):
        """
        获取缓冲区中的所有样本
        :param shuffle: 是否打乱样本顺序 (默认为True)
        :return: (样本张量, 标签张量)
        """
        if len(self.buffer) == 0:
            # 根据输入通道数动态调整维度
            return (
                torch.empty(0, self.input_channels, 32, 32, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device)
            )

        # 如果需要打乱顺序
        if shuffle:
            shuffled_buffer = self.buffer[:]
            random.shuffle(shuffled_buffer)
            samples, labels = zip(*shuffled_buffer)
        else:
            samples, labels = zip(*self.buffer)

        # 转换为tensor
        samples_array = np.array(samples)
        samples = torch.tensor(samples_array, dtype=torch.float32, device=self.device)
        
        # 确保数据形状正确
        if len(samples.shape) == 4:  # [N, C, H, W] - 正确形状
            # 确保通道数正确
            if samples.shape[1] != self.input_channels:
                if self.input_channels == 3 and samples.shape[1] == 1:
                    # 单通道转三通道
                    samples = samples.repeat(1, 3, 1, 1)
                elif self.input_channels == 1 and samples.shape[1] == 3:
                    # 三通道转单通道
                    samples = samples.mean(dim=1, keepdim=True)
        elif len(samples.shape) == 3:  # [N, H, W] - 单通道情况
            samples = samples.unsqueeze(1)  # 添加通道维度 -> [N, 1, H, W]
            if self.input_channels == 3:
                # 单通道转三通道
                samples = samples.repeat(1, 3, 1, 1)
        
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        return samples, labels

    def update_samples(self, new_samples, new_labels):
        combined = list(zip(new_samples, new_labels))
        self.buffer = combined[:self.size]

    def __len__(self):
        return len(self.buffer)

    def get_class_distribution(self):
        """获取类别分布统计"""
        if not self.buffer:
            return {}
        _, labels = zip(*self.buffer)
        return dict(zip(*np.unique(labels, return_counts=True)))

    def get_statistics(self):
        """
        返回内存统计摘要
        """
        class_dist = self.get_class_distribution()
        total = len(self)

        stats = {
            "total_samples": total,
            "class_distribution": class_dist,
            "num_classes": len(class_dist),
            "min_samples": min(class_dist.values()) if class_dist else 0,
            "max_samples": max(class_dist.values()) if class_dist else 0,
            "diversity_index": len(class_dist) / total if total > 0 else 0
        }
        return stats

    def print_statistics(self):
        """
        打印内存统计摘要
        """
        stats = self.get_statistics()
        print("\n内存统计摘要:")
        print(f"总样本数: {stats['total_samples']}")
        print(f"类别数量: {stats['num_classes']}")
        print(f"最小类别样本数: {stats['min_samples']}")
        print(f"最大类别样本数: {stats['max_samples']}")
        print(f"多样性指数: {stats['diversity_index']:.4f}")
        if stats['class_distribution']:
            print("详细分布:")
            for cls, count in sorted(stats['class_distribution'].items()):
                print(f"  类别 {cls}: {count} 样本")


class DualMemoryBuffer:
    """Short/long memory buffer used by SDM-CL.

    Short memory is a FIFO window for the current stream distribution. Long
    memory stores high-information samples selected by the current expert.
    """

    def __init__(self, short_size=200, long_size=500, input_channels=1, img_size=32):
        self.short_size = int(short_size)
        self.long_size = int(long_size)
        self.input_channels = input_channels
        self.img_size = img_size
        self.short_buffer = []
        self.long_buffer = []
        self.last_distance = None
        self.last_short_phi = None
        self.last_long_phi = None

    @property
    def size(self):
        return self.short_size + self.long_size

    def __len__(self):
        return len(self.short_buffer) + len(self.long_buffer)

    def _process_sample(self, sample):
        sample_tensor = torch.as_tensor(sample, dtype=torch.float32).detach().cpu()

        if sample_tensor.dim() == 2:
            sample_tensor = sample_tensor.unsqueeze(0)
        elif sample_tensor.dim() == 4 and sample_tensor.size(0) == 1:
            sample_tensor = sample_tensor.squeeze(0)

        if sample_tensor.dim() != 3:
            raise ValueError(f"Expected sample shape [C,H,W] or [H,W], got {tuple(sample_tensor.shape)}")

        if sample_tensor.shape[0] != self.input_channels:
            if self.input_channels == 3 and sample_tensor.shape[0] == 1:
                sample_tensor = sample_tensor.repeat(3, 1, 1)
            elif self.input_channels == 1 and sample_tensor.shape[0] == 3:
                sample_tensor = sample_tensor.mean(dim=0, keepdim=True)
            else:
                raise ValueError(
                    f"Cannot adapt {sample_tensor.shape[0]} channels to {self.input_channels}"
                )

        return sample_tensor.numpy()

    def _make_entries(self, samples, labels, scores=None):
        if isinstance(samples, torch.Tensor):
            samples_iter = samples.detach().cpu()
        else:
            samples_iter = samples

        if isinstance(labels, torch.Tensor):
            labels_iter = labels.detach().cpu().tolist()
        else:
            labels_iter = list(labels)

        if scores is None:
            scores_iter = [None] * len(labels_iter)
        elif isinstance(scores, torch.Tensor):
            scores_iter = scores.detach().cpu().tolist()
        else:
            scores_iter = list(scores)

        entries = []
        for sample, label, score in zip(samples_iter, labels_iter, scores_iter):
            entries.append((
                self._process_sample(sample),
                int(label),
                None if score is None else float(score),
            ))
        return entries

    @property
    def long_full(self):
        return len(self.long_buffer) >= self.long_size

    @property
    def long_remaining(self):
        return max(0, self.long_size - len(self.long_buffer))

    def add_long_samples(self, samples, labels):
        if self.long_remaining <= 0:
            return 0

        entries = self._make_entries(samples, labels)
        accepted = entries[:self.long_remaining]
        self.long_buffer.extend(accepted)
        return len(accepted)

    def add_short_samples(self, samples, labels):
        entries = self._make_entries(samples, labels)
        self.short_buffer.extend(entries)

        overflow = len(self.short_buffer) - self.short_size
        if overflow > 0:
            self.short_buffer = self.short_buffer[overflow:]

        return self.short_ready

    @property
    def short_ready(self):
        return len(self.short_buffer) >= self.short_size

    def clear_short(self):
        self.short_buffer = []

    def clear_long(self):
        self.long_buffer = []

    def reset(self):
        self.clear_short()
        self.clear_long()
        self.last_distance = None
        self.last_short_phi = None
        self.last_long_phi = None

    def _entries_to_tensors(self, entries, shuffle=True):
        if not entries:
            return (
                torch.empty(0, self.input_channels, self.img_size, self.img_size),
                torch.empty(0, dtype=torch.long),
            )

        selected = entries[:]
        if shuffle:
            random.shuffle(selected)

        samples, labels, _ = zip(*selected)
        samples = torch.tensor(np.array(samples), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return samples, labels

    def get_short_samples(self, shuffle=True):
        return self._entries_to_tensors(self.short_buffer, shuffle=shuffle)

    def get_long_samples(self, shuffle=True):
        return self._entries_to_tensors(self.long_buffer, shuffle=shuffle)

    def get_training_samples(self, shuffle=True):
        return self._entries_to_tensors(self.long_buffer + self.short_buffer, shuffle=shuffle)

    def sample_long_samples(self, num_samples):
        if not self.long_buffer:
            return (
                torch.empty(0, self.input_channels, self.img_size, self.img_size),
                torch.empty(0, dtype=torch.long),
            )

        if len(self.long_buffer) >= num_samples:
            entries = random.sample(self.long_buffer, num_samples)
        else:
            entries = random.choices(self.long_buffer, k=num_samples)
        return self._entries_to_tensors(entries, shuffle=False)

    def _prepare_for_component(self, samples, component):
        samples = samples.float().to(component.device)
        if samples.dim() == 3:
            samples = samples.unsqueeze(1)

        if component.input_channels == 1 and samples.shape[1] == 3:
            samples = samples.mean(dim=1, keepdim=True)
        elif component.input_channels == 3 and samples.shape[1] == 1:
            samples = samples.repeat(1, 3, 1, 1)

        return samples.reshape(
            samples.size(0),
            component.input_channels,
            self.img_size,
            self.img_size,
        )

    @staticmethod
    def _psp_filter(spike_profile, tau_syn):
        syn = torch.zeros(
            spike_profile.size(0),
            dtype=spike_profile.dtype,
            device=spike_profile.device,
        )
        traces = []
        for t in range(spike_profile.size(-1)):
            syn = syn + (spike_profile[:, t] - syn) / tau_syn
            traces.append(syn)
        return torch.stack(traces, dim=-1)

    def _encode_phi(self, component, samples, n_steps, tau_syn=10.0, batch_size=64):
        if samples.numel() == 0:
            return None

        was_training = component.vae.training
        component.vae.eval()
        profile_sum = None
        total = 0

        with torch.no_grad():
            for start in range(0, samples.size(0), batch_size):
                batch = samples[start:start + batch_size]
                batch = self._prepare_for_component(batch, component)
                spike_input = batch.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)
                sampled_z, _, _ = component.vae.encode(spike_input)
                sampled_z = sampled_z.float()

                if sampled_z.dim() == 5:
                    batch_profile = sampled_z.mean(dim=(0, 2, 3))
                elif sampled_z.dim() == 3:
                    batch_profile = sampled_z.mean(dim=0)
                elif sampled_z.dim() == 2:
                    batch_profile = sampled_z.mean(dim=0, keepdim=False).unsqueeze(-1)
                else:
                    raise ValueError(f"Unexpected spike code shape: {tuple(sampled_z.shape)}")

                current = batch.size(0)
                if profile_sum is None:
                    profile_sum = batch_profile * current
                else:
                    profile_sum = profile_sum + batch_profile * current
                total += current

        if was_training:
            component.vae.train()

        mean_profile = profile_sum / max(total, 1)
        psp_trace = self._psp_filter(mean_profile, tau_syn=tau_syn)
        return psp_trace.mean(dim=-1)

    def compute_psp_distance(self, component, n_steps, tau_syn=10.0, batch_size=64):
        short_samples, _ = self.get_short_samples(shuffle=False)
        long_samples, _ = self.get_long_samples(shuffle=False)

        if short_samples.numel() == 0 or long_samples.numel() == 0:
            self.last_distance = 0.0
            self.last_short_phi = None
            self.last_long_phi = None
            return 0.0

        short_phi = self._encode_phi(component, short_samples, n_steps, tau_syn, batch_size)
        long_phi = self._encode_phi(component, long_samples, n_steps, tau_syn, batch_size)
        distance = torch.norm(short_phi - long_phi, p=2).item()

        self.last_distance = distance
        self.last_short_phi = short_phi.detach().cpu()
        self.last_long_phi = long_phi.detach().cpu()
        return distance

    def compute_fsvae_mmd(self, component, batch_size=64):
        short_samples, _ = self.get_short_samples(shuffle=False)
        if short_samples.numel() == 0 or len(self.long_buffer) == 0:
            self.last_distance = 0.0
            self.last_short_phi = None
            self.last_long_phi = None
            return 0.0

        long_samples, _ = self.sample_long_samples(short_samples.size(0))
        mmd = component.compute_memory_mmd(
            short_samples,
            long_samples,
            batch_size=batch_size,
        ).item()

        self.last_distance = mmd
        self.last_short_phi = None
        self.last_long_phi = None
        return mmd

    def consolidate(self, component, batch_size=64):
        if not self.short_buffer:
            return 0

        short_samples, short_labels = self.get_short_samples(shuffle=False)
        short_scores = component.compute_information_scores(short_samples, batch_size=batch_size)

        scored_entries = []
        for idx, score in enumerate(short_scores.tolist()):
            scored_entries.append((
                "short",
                short_samples[idx].numpy(),
                int(short_labels[idx].item()),
                float(score),
            ))

        if self.long_buffer:
            long_samples, long_labels = self.get_long_samples(shuffle=False)
            long_scores = component.compute_information_scores(long_samples, batch_size=batch_size)
            for idx, score in enumerate(long_scores.tolist()):
                scored_entries.append((
                    "long",
                    long_samples[idx].numpy(),
                    int(long_labels[idx].item()),
                    float(score),
                ))

        scored_entries.sort(key=lambda item: item[3], reverse=True)
        selected = scored_entries[:self.long_size]
        self.long_buffer = [
            (np.array(sample, dtype=np.float32), label, score)
            for _, sample, label, score in selected
        ]
        return sum(1 for source, _, _, _ in selected if source == "short")

    def get_statistics(self):
        def summarize_entries(entries):
            labels = [label for _, label, _ in entries]
            scores = [score for _, _, score in entries if score is not None]
            class_distribution = {}
            score_summary = None

            if labels:
                unique, counts = np.unique(labels, return_counts=True)
                class_distribution = dict(zip(unique.tolist(), counts.tolist()))

            if scores:
                score_array = np.array(scores, dtype=np.float32)
                score_summary = {
                    "min": float(score_array.min()),
                    "mean": float(score_array.mean()),
                    "max": float(score_array.max()),
                }

            return {
                "samples": len(entries),
                "class_distribution": class_distribution,
                "score_summary": score_summary,
            }

        short_stats = summarize_entries(self.short_buffer)
        long_stats = summarize_entries(self.long_buffer)

        return {
            "short_samples": len(self.short_buffer),
            "long_samples": len(self.long_buffer),
            "total_samples": len(self),
            "short": short_stats,
            "long": long_stats,
            "class_distribution": summarize_entries(self.short_buffer + self.long_buffer)["class_distribution"],
            "last_distance": self.last_distance,
        }

    def print_statistics(self):
        stats = self.get_statistics()
        print("========== SDM memory state ==========")
        print(f"Short memory: {stats['short_samples']}/{self.short_size}")
        print(f"  classes: {stats['short']['class_distribution']}")
        print(f"Long memory:  {stats['long_samples']}/{self.long_size}")
        print(f"  classes: {stats['long']['class_distribution']}")
        if stats["long"]["score_summary"] is not None:
            score = stats["long"]["score_summary"]
            print(
                "  information score: "
                f"min={score['min']:.6f}, mean={score['mean']:.6f}, max={score['max']:.6f}"
            )
        print(f"Total memory: {stats['total_samples']}/{self.size}")
        if stats["last_distance"] is not None:
            print(f"Last FSVAE-MMD distance: {stats['last_distance']:.6f}")
        print("======================================")


class PermutedMNIST(torch.utils.data.Dataset):
    def __init__(self, base_dataset, perm):
        """
        base_dataset: 通常是 torchvision.datasets.MNIST 或 Subset(MNIST, indices_in_full)
        perm: 长度为 32*32 的 torch.LongTensor 或 torch.Tensor 索引
        """
        self.dataset = base_dataset
        self.perm = perm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]       # img: [1,32,32]
        img = img.view(-1)[self.perm]         # 展平并按 perm 重排（长度 1024）
        img = img.view(1, 32, 32)             # reshape 回 (1,32,32)
        return img, target
