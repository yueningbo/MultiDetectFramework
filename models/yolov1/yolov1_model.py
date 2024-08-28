import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, num_bounding_boxes, num_classes):
        super().__init__()

        # 卷积层部分
        self.conv_layers = nn.Sequential(
            # 输入图像大小为 448x448
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 输出尺寸: 448x448
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 输出尺寸: 224x224

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输出尺寸: 224x224
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 输出尺寸: 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输出尺寸: 112x112
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 输出尺寸: 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输出尺寸: 56x56
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 输出尺寸: 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 输出尺寸: 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 输出尺寸: 14x14

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 输出尺寸: 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 输出尺寸: 7x7
        )

        # 全连接层部分
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * (5 * num_bounding_boxes + num_classes))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)
