import json

import torch
from thop import profile
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1: Tensor of shape (N, 4) with format [x_center, y_center, width, height]
    - box2: Tensor of shape (M, 4) with format [x_center, y_center, width, height]

    Returns:
    - Tensor of shape (N, M) representing IoU between each pair of boxes from box1 and box2.
    """
    box1_x1, box1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    box1_x2, box1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    box2_x1, box2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    box2_x2, box2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    # Compute intersection coordinates
    inter_x1 = torch.max(box1_x1.unsqueeze(1), box2_x1.unsqueeze(0))
    inter_y1 = torch.max(box1_y1.unsqueeze(1), box2_y1.unsqueeze(0))
    inter_x2 = torch.min(box1_x2.unsqueeze(1), box2_x2.unsqueeze(0))
    inter_y2 = torch.min(box1_y2.unsqueeze(1), box2_y2.unsqueeze(0))

    # Compute intersection area
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height

    # Compute areas of each box
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Compute IoU
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
    iou = inter_area / (union_area + 1e-6)  # Avoid division by zero

    return iou


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


def nms(boxes, scores, threshold):
    """
    非极大值抑制算法

    Args:
        boxes (Tensor): 边界框 [num_boxes, 5]，每个框 [x_center, y_center, w, h, conf]
        scores (Tensor): 置信度
        threshold (float): NMS 阈值

    Returns:
        list: 保留的框的索引列表
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while sorted_indices.size(0) > 0:
        i = sorted_indices[0]
        keep.append(i.item())

        if sorted_indices.size(0) == 1:
            break

        xx1 = torch.max(x1[sorted_indices[1:]], x1[i])
        yy1 = torch.max(y1[sorted_indices[1:]], y1[i])
        xx2 = torch.min(x2[sorted_indices[1:]], x2[i])
        yy2 = torch.min(y2[sorted_indices[1:]], y2[i])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        union = areas[sorted_indices[1:]] + areas[i] - inter
        iou = inter / (union + 1e-6)

        sorted_indices = sorted_indices[1:][iou <= threshold]

    return keep
