import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .classifier import build_classifier
from .vae_fsvae import SpikingFSVAE


class Component:
    """One SDM-CL expert: a spiking VAE plus a spiking classifier."""

    def __init__(
        self,
        gan_z_dim=128,
        learning_rate=0.003,
        beta1=0.5,
        batch_size=512,
        n_steps=8,
        input_channels=1,
        num_classes=10,
        img_size=32,
        classifier_type="ann",
    ):
        self.n_steps = n_steps
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.classifier_type = classifier_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_frozen = False

        self.vae = SpikingFSVAE(in_channels=input_channels, n_steps=n_steps).to(self.device)
        params = list(self.vae.named_parameters())
        sample_layer_lr_times = 10
        sample_params = [p for n, p in params if "sample_layer" in n]
        base_params = [p for n, p in params if "sample_layer" not in n]
        param_group = []
        if sample_params:
            param_group.append({
                "params": sample_params,
                "weight_decay": 0.001,
                "lr": learning_rate * sample_layer_lr_times,
            })
        if base_params:
            param_group.append({
                "params": base_params,
                "weight_decay": 0.001,
                "lr": learning_rate,
            })
        self.vae_optimizer = torch.optim.AdamW(
            param_group,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.001,
        )

        self.classifier = build_classifier(
            classifier_type=classifier_type,
            input_channels=input_channels,
            num_classes=num_classes,
            n_steps=n_steps,
            img_size=img_size,
        ).to(self.device)
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate * 10,
            betas=(beta1, 0.999),
        )

        self.batch_size = batch_size
        self.z_dim = gan_z_dim

    def _prepare_input(self, x):
        x = x.float().to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if self.input_channels == 1 and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        elif self.input_channels == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return x.reshape(x.size(0), self.input_channels, self.img_size, self.img_size)

    def _to_spike_input(self, x):
        x = self._prepare_input(x)
        return x, x.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_steps)

    def train_vae(self, x):
        if self.is_frozen:
            return 0.0

        self.vae.train()
        x, spike_input = self._to_spike_input(x)
        x_recon, q_z, p_z, _ = self.vae(spike_input, scheduled=True)
        loss = self.vae.loss_function_mmd(x, x_recon, q_z, p_z)

        self.vae_optimizer.zero_grad()
        loss["loss"].backward()
        self.vae_optimizer.step()
        return loss["loss"].item()

    def get_mmd_loss(self, x):
        x, spike_input = self._to_spike_input(x)
        return self.vae.get_check_mmd_loss(spike_input)

    def train_classifier(self, x, labels):
        if self.is_frozen:
            return 0.0

        self.classifier.train()
        x = self._prepare_input(x)
        labels = labels.long().to(self.device)
        outputs = self.classifier(x)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        return loss.item()

    def compute_information_scores(self, x, batch_size=None):
        """Negative-ELBO proxy used by SDM-CL memory consolidation."""
        if batch_size is None:
            batch_size = self.batch_size

        scores = []
        was_training = self.vae.training
        self.vae.eval()

        with torch.no_grad():
            for start in range(0, x.size(0), batch_size):
                batch = x[start:start + batch_size]
                prepared, spike_input = self._to_spike_input(batch)
                x_recon, q_z, p_z, _ = self.vae(spike_input, scheduled=True)

                if hasattr(self.vae, "batch_loss_function_mmd"):
                    loss = self.vae.batch_loss_function_mmd(
                        prepared,
                        x_recon,
                        q_z,
                        p_z,
                    )["loss"]
                    if loss.dim() == 0:
                        loss = loss.repeat(prepared.size(0))
                else:
                    loss = F.mse_loss(x_recon, prepared, reduction="none")
                    loss = loss.view(loss.size(0), -1).mean(dim=1)

                scores.append(loss.detach().cpu())

        if was_training:
            self.vae.train()

        return torch.cat(scores, dim=0) if scores else torch.empty(0)

    def encode_mmd_features(self, x, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        features = []
        was_training = self.vae.training
        self.vae.eval()

        with torch.no_grad():
            for start in range(0, x.size(0), batch_size):
                batch = x[start:start + batch_size]
                _, spike_input = self._to_spike_input(batch)
                _, q_z, _ = self.vae.encode(spike_input)
                features.append(torch.mean(q_z, dim=2).detach())

        if was_training:
            self.vae.train()

        return torch.cat(features, dim=0) if features else torch.empty(0, device=self.device)

    def compute_memory_mmd(self, source_samples, target_samples, batch_size=None):
        source_z = self.encode_mmd_features(source_samples, batch_size=batch_size)
        target_z = self.encode_mmd_features(target_samples, batch_size=batch_size)
        return self.vae.latent_mmd(source_z, target_z)

    def test_classifier(self, x):
        self.classifier.eval()
        x = self._prepare_input(x)
        with torch.no_grad():
            outputs = self.classifier(x)
            predicted = torch.argmax(outputs, dim=1)
        return predicted

    def freeze(self):
        self.is_frozen = True
        self.eval()
        for module in (self.vae, self.classifier):
            for param in module.parameters():
                param.requires_grad_(False)

    def to(self, device):
        self.vae = self.vae.to(device)
        self.classifier = self.classifier.to(device)
        self.device = device
        return self

    def eval(self):
        self.vae.eval()
        self.classifier.eval()

    def train(self):
        if not self.is_frozen:
            self.vae.train()
            self.classifier.train()
