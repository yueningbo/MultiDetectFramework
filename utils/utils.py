import json

import torch
from thop import profile
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_iou(pred_boxes, target_boxes):
    """
    计算预测框和目标框之间的IoU。

    Args:
        pred_boxes: (Tensor) 预测框，形状为 [B, 4]，其中 B 是预测框的数量。
        target_boxes: (Tensor) 目标框，形状为 [1, 4]。

    Returns:
        iou: (Tensor) IoU 值，形状为 [B, 1]。
    """
    # 计算交集的左上角和右下角坐标
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # 计算交集的宽度和高度
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算交集面积
    inter_area = inter_w * inter_h

    # 计算预测框和目标框的面积
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    # 计算并集面积
    union_area = pred_area + target_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou.unsqueeze(1)  # 返回形状为 [B, 1] 的IoU张量


def print_model_flops(model, input_size, device):
    """
    Print the FLOPs (Floating Point Operations) and number of parameters of the model.

    Parameters:
    - model: The PyTorch model to evaluate.
    - input_size: Tuple representing the size of the input tensor, e.g., (3, 224, 224).
    - device: The device where the model is located, e.g., 'cuda:0' or 'cpu'.
    """
    dummy_input = torch.randn(1, *input_size).to(device)

    # Calculate FLOPs and number of parameters
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    # Format FLOPs and parameters for readability
    macs_readable = f"{macs / 1e9:.2f} GFLOPs"
    params_readable = f"{params / 1e6:.2f} M"

    logging.info(f"Model FLOPs: {macs_readable}")
    logging.info(f"Model Parameters: {params_readable}")


def denormalize(img):
    """
    Reverse the normalization of an image.

    img (Tensor): The input image tensor with normalized values.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Reverse the normalization
    img = img * std + mean

    return img


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    logging.info(f'Configuration: {config}')
    return config


def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from format [xmin, ymin, w, h]
    to corner format [xmin, ymin, xmax, ymax].

    Args:
        boxes (Tensor): Tensor of shape [num_boxes, 4] where each row represents
                        a bounding box in format [xmin, ymin, w, h].

    Returns:
        Tensor: Tensor of shape [num_boxes, 4] where each row represents
                the bounding box in corner format [xmin, ymin, xmax, ymax].
    """
    # Compute corner coordinates directly using tensor operations
    corners = torch.cat((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]), dim=1)
    return corners


def nms(boxes: torch.Tensor, scores, iou_threshold):
    """
    执行非极大值抑制（NMS）。

    参数:
    - boxes: 形状为 (N, 4) 的张量，其中 N 是框的数量，每行是一个 [x1, y1, x2, y2] 的边界框。
    - scores: 形状为 (N,) 的张量，包含每个框的得分。
    - iou_threshold: 用于决定何时认为两个框重叠的 IoU 阈值。

    返回:
    - keep: 一个包含应保留的框的索引的张量。
    """
    # Sort the bounding boxes by scores in descending order
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        iou = compute_iou(boxes[order[1:]], boxes[i].unsqueeze(0))

        order = order[1:][iou.squeeze() <= iou_threshold]

    return torch.tensor(keep, dtype=torch.int64)


if __name__ == '__main__':
    # 示例使用
    boxes = torch.tensor([[100, 100, 210, 210], [105, 105, 215, 215], [300, 300, 400, 400]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.75, 0.6], dtype=torch.float32)
    iou_threshold = 0.5

    from torchvision.ops import nms as tnms

    assert tnms(boxes, scores, iou_threshold) == nms(boxes, scores, iou_threshold)
