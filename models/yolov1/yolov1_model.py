import logging

import torch
import torch.nn as nn
import torchvision.models as models


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20, pretrained_weights_path=None):
        super().__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes
        self.C = C  # Number of classes

        # Load pretrained MobileNetV2 as the backbone
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        # Use MobileNetV2 up to the penultimate layer
        self.backbone = nn.Sequential(
            *list(mobilenet_v2.features.children()),  # Keep all layers
            nn.AdaptiveAvgPool2d((self.S, self.S))  # Adjust to SxS grid
        )

        # Reduce the number of output channels to make it simpler
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),  # Reducing output channels
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, (self.C + self.B * 5), kernel_size=1)
        )

        if pretrained_weights_path:
            self._load_pretrained_weights(pretrained_weights_path)
        else:
            logging.info('Model weights init.')
            self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_layers(x)

        # [batch, C, H, W] -> [batch, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize weights using He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
