import torch
from thop import profile
import logging


def bbox_iou(pred_boxes, true_boxes):
    """
    Compute the Intersection over Union (IoU) of two sets of bounding boxes.
    pred_boxes and true_boxes are of shape (N, 4) where N is the number of boxes.
    Returns an IoU matrix of shape (N, M) where N is the number of predicted boxes and M is the number of true boxes.
    """
    # Calculate area of prediction and ground truth boxes
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

    # Calculate intersection areas
    x1 = torch.max(pred_boxes[:, 0].unsqueeze(1), true_boxes[:, 0].unsqueeze(0))
    y1 = torch.max(pred_boxes[:, 1].unsqueeze(1), true_boxes[:, 1].unsqueeze(0))
    x2 = torch.min(pred_boxes[:, 2].unsqueeze(1), true_boxes[:, 2].unsqueeze(0))
    y2 = torch.min(pred_boxes[:, 3].unsqueeze(1), true_boxes[:, 3].unsqueeze(0))

    inter_area = (x2 - x1) * (y2 - y1)
    union_area = pred_area.unsqueeze(1) + true_area.unsqueeze(0) - inter_area

    iou = inter_area / torch.clamp(union_area, min=1e-6)
    return iou


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
