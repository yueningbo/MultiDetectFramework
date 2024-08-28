import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes
        self.C = C  # Number of classes

        # Define the convolutional layers (backbone)
        self.conv_layers = nn.Sequential(
            self._conv_block(3, 64, kernel_size=7, stride=2, padding=3, max_pool=True),
            self._conv_block(64, 192, kernel_size=3, padding=1, max_pool=True),
            self._conv_block(192, 128, kernel_size=1),
            self._conv_block(128, 256, kernel_size=3, padding=1),
            self._conv_block(256, 256, kernel_size=1),
            self._conv_block(256, 512, kernel_size=3, padding=1, max_pool=True),
            self._repeat_conv_blocks(512, 256, 512, num_repeats=4),
            self._conv_block(512, 1024, kernel_size=3, padding=1, max_pool=True),
            self._repeat_conv_blocks(1024, 512, 1024, num_repeats=2),
            self._conv_block(1024, 1024, kernel_size=3, padding=1),
            self._conv_block(1024, 1024, kernel_size=3, stride=2, padding=1),
            self._repeat_conv_blocks(1024, 1024, 1024, num_repeats=2)
        )

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0, max_pool=False):
        """Helper function to create a convolutional block."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        ]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _repeat_conv_blocks(self, in_channels, mid_channels, out_channels, num_repeats):
        """Helper function to repeat conv blocks with a bottleneck structure."""
        layers = []
        for _ in range(num_repeats):
            layers.append(self._conv_block(in_channels, mid_channels, kernel_size=1))
            layers.append(self._conv_block(mid_channels, out_channels, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x

    def load_weights(self, weights_path):
        """Load weights from a file."""
        self.load_state_dict(torch.load(weights_path))

    def save_weights(self, save_path):
        """Save weights to a file."""
        torch.save(self.state_dict(), save_path)
