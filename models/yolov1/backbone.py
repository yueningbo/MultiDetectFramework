from torch import nn
from models.yolov1.basic import ConvBlock


def repeat(block, num_repeats):
    """Helper function to repeat a block multiple times."""
    return nn.Sequential(*[block() for _ in range(num_repeats)])


class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()

        # 定义网络层
        self.layers = nn.Sequential(
            # 第一层：输入 448x448x3 -> 输出 224x224x64
            ConvBlock(3, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),  # 224x224x64 -> 112x112x64

            # 第二层：输入 112x112x64 -> 输出 112x112x192
            ConvBlock(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 112x112x192 -> 56x56x192

            # 第三层：输入 56x56x192 -> 输出 56x56x512
            ConvBlock(192, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 56x56x512 -> 28x28x512

            # 第四层：输入 28x28x512 -> 输出 28x28x1024
            repeat(lambda: nn.Sequential(
                ConvBlock(512, 256, 1, 1, 0),
                ConvBlock(256, 512, 3, 1, 1),
            ), 4),
            ConvBlock(512, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 28x28x1024 -> 14x14x1024

            # 第五层：输入 14x14x1024 -> 输出 7x7x1024
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 2, 1),  # 14x14x1024 -> 7x7x1024

            # 第六层：输入 7x7x1024 -> 输出 7x7x1024
            repeat(lambda: ConvBlock(1024, 1024, 3, 1, 1), 2)
        )

    def forward(self, x):
        return self.layers(x)


print(Darknet())
