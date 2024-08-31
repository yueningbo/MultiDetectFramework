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
            # 第一层：输入 448x448x3 -> 输出 224x224x32
            ConvBlock(3, 32, 7, 2, 3),
            nn.MaxPool2d(2, 2),  # 224x224x32 -> 112x112x32

            # 第二层：输入 112x112x32 -> 输出 112x112x64
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 112x112x64 -> 56x56x64

            # 第三层：输入 56x56x64 -> 输出 56x56x128
            ConvBlock(64, 64, 1, 1, 0),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 56x56x256 -> 28x28x256

            # 第四层：输入 28x28x256 -> 输出 28x28x512
            repeat(lambda: nn.Sequential(
                ConvBlock(256, 128, 1, 1, 0),
                ConvBlock(128, 256, 3, 1, 1),
            ), 2),
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 28x28x512 -> 14x14x512

            # 第五层：输入 14x14x512 -> 输出 7x7x512
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 512, 3, 1, 1),
            ConvBlock(512, 512, 3, 2, 1),  # 14x14x512 -> 7x7x512

            # 第六层：输入 7x7x512 -> 输出 7x7x512
            repeat(lambda: ConvBlock(512, 512, 3, 1, 1), 2)
        )

    def forward(self, x):
        return self.layers(x)
