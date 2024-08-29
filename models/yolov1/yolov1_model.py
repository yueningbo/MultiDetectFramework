import torch
import torch.nn as nn
import torchvision.models as models


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_layers(x)

        # [batch, C, H, W] -> [batch, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


if __name__ == '__main__':
    # Example usage
    model = YOLOv1(S=7, B=2, C=20)
    print(model)
