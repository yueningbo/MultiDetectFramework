import torch
from thop import profile
import logging


def bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x1g, y1g, x2g, y2g = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter = (torch.min(x2, x2g) - torch.max(x1, x1g)) * (torch.min(y2, y2g) - torch.max(y1, y1g)).clamp(min=0)
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter
    IoU = inter / union
    return IoU


def print_model_flops(model, input_size):
    """
    打印模型的 FLOPs 和参数量。

    参数:
    - model: 要评估的 PyTorch 模型。
    - input_size: 模型输入的尺寸，例如 (3, 224, 224)。
    """
    # 创建一个假的输入张量
    dummy_input = torch.randn(1, *input_size)

    # 计算 FLOPs 和参数量
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 转换为可读的格式
    macs_readable = f"{macs / 1e9:.2f} GFLOPs"
    params_readable = f"{params / 1e6:.2f} M"

    logging.info(f"Model FLOPs: {macs_readable}")
    logging.info(f"Model Parameters: {params_readable}")
