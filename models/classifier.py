import torch
import torch.nn as nn
import torchvision.models as models

from fsvae_models.snn_layers import LIFSpike, tdBatchNorm, tdConv, tdLinear


class Classifier(nn.Module):
    """Legacy ANN ResNet classifier kept for old scripts that import it directly."""

    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.model = models.resnet18(pretrained=False)

        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ANNClassifier(nn.Module):
    """Fast CNN classifier used by default while the VAE remains fully spiking."""

    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MLPClassifier(nn.Module):
    """Fast ANN MLP for MNIST-like 32x32 grayscale inputs."""

    def __init__(self, input_channels=1, num_classes=10, img_size=32):
        super().__init__()
        in_features = input_channels * img_size * img_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class SpikingClassifier(nn.Module):
    """Compact fully spiking classifier for 32x32 image streams."""

    def __init__(self, input_channels=1, num_classes=10, n_steps=16):
        super().__init__()
        self.n_steps = n_steps

        self.features = nn.Sequential(
            tdConv(
                input_channels,
                32,
                kernel_size=3,
                padding=1,
                bias=True,
                bn=tdBatchNorm(32),
                spike=LIFSpike(),
                is_first_conv=True,
            ),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            tdConv(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=True,
                bn=tdBatchNorm(64),
                spike=LIFSpike(),
            ),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
        )
        self.fc1 = tdLinear(
            64 * 8 * 8,
            128,
            bias=True,
            bn=tdBatchNorm(128),
            spike=LIFSpike(),
        )
        self.fc2 = tdLinear(128, num_classes, bias=True, bn=None, spike=None)

        coef = torch.pow(0.8, torch.arange(self.n_steps - 1, -1, -1).float())
        self.register_buffer("readout_coef", coef.view(1, 1, self.n_steps))

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_steps)

        x = self.features(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sum(x * self.readout_coef, dim=-1) / self.readout_coef.sum()


class MNISTClassifier(SpikingClassifier):
    """Backward-compatible name used by Component."""

    def __init__(self, input_channels=1, num_classes=10, n_steps=16):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            n_steps=n_steps,
        )


def build_classifier(classifier_type, input_channels=1, num_classes=10, n_steps=16, img_size=32):
    classifier_type = classifier_type.lower()
    if classifier_type == "ann":
        if input_channels == 1:
            return MLPClassifier(
                input_channels=input_channels,
                num_classes=num_classes,
                img_size=img_size,
            )
        return ANNClassifier(input_channels=input_channels, num_classes=num_classes)
    if classifier_type == "snn":
        return SpikingClassifier(
            input_channels=input_channels,
            num_classes=num_classes,
            n_steps=n_steps,
        )
    if classifier_type == "resnet18":
        return Classifier(num_classes=num_classes, input_channels=input_channels)
    raise ValueError(f"Unsupported classifier_type: {classifier_type}")
