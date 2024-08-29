import torch
from thop import profile
import logging


def compute_iou(box1, box2):
    """
    计算两个边界框的IoU
    box1, box2: [x_center, y_center, width, height]
    """
    # 将中心点形式的框转换为左上角和右下角
    box1_x1, box1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    box1_x2, box1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    box2_x1, box2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    box2_x2, box2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    # 计算相交区域的坐标
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    # 计算相交区域的面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算两个框的面积
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # 计算IoU
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)  # 防止除零
    return iou


def print_model_flops(model, input_size, device):
    """
    打印模型的 FLOPs 和参数量。

    参数:
    - model: 要评估的 PyTorch 模型。
    - input_size: 模型输入的尺寸，例如 (3, 224, 224)。
    """
    # 创建一个假的输入张量
    dummy_input = torch.randn(1, *input_size).to(device)

    # 计算 FLOPs 和参数量
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 转换为可读的格式
    macs_readable = f"{macs / 1e9:.2f} GFLOPs"
    params_readable = f"{params / 1e6:.2f} M"

    logging.info(f"Model FLOPs: {macs_readable}")
    logging.info(f"Model Parameters: {params_readable}")
