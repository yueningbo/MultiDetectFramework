import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class YOLOv1Transform:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, image, annotations):
        # 获取目标尺寸
        target_size = self.size
        # 计算需要填充的尺寸
        ratio = float(min(target_size[0] / image.width, target_size[1] / image.height))
        new_size = (int(image.width * ratio), int(image.height * ratio))
        # 缩放图像
        image = TF.resize(image, new_size)
        # 计算填充的尺寸
        pad = (target_size[0] - new_size[0]) // 2
        pad_top = pad
        pad_bottom = target_size[0] - new_size[0] - pad
        pad_left = (target_size[1] - new_size[1]) // 2
        pad_right = target_size[1] - new_size[1] - pad_left
        # 填充图像
        image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), 0)

        # 调整注释
        img_width, img_height = image.size
        scale = torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)
        annotations[:, [1, 2, 3, 4]] *= scale

        # 转换图像为张量
        image = TF.to_tensor(image)

        return image, annotations
