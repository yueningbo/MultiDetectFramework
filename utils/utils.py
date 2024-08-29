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


def center_to_corners(boxes):
    """
    Convert bounding boxes from center format [x_center, y_center, w, h]
    to corner format [xmin, ymin, xmax, ymax].

    Args:
        boxes (Tensor): Tensor of shape [num_boxes, 4] where each row represents
                        a bounding box in center format [x_center, y_center, w, h].

    Returns:
        Tensor: Tensor of shape [num_boxes, 4] where each row represents
                the bounding box in corner format [xmin, ymin, xmax, ymax].
    """
    # Extract bounding box parameters
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Compute corner coordinates
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2

    # Stack the coordinates into a single tensor
    corners = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    return corners


def nms(boxes, scores, iou_threshold, format="XYHW"):
    """
    执行非极大值抑制（NMS）。

    参数:
    - boxes: 形状为 (N, 4) 的张量，其中 N 是框的数量，每行是一个 [x1, y1, x2, y2] 的边界框。
    - scores: 形状为 (N,) 的张量，包含每个框的得分。
    - iou_threshold: 用于决定何时认为两个框重叠的 IoU 阈值。

    返回:
    - keep: 一个包含应保留的框的索引的张量。
    """
    indices = torch.argsort(scores, descending=True)
    keep = []

    while indices.numel() > 0:
        current = indices[0]
        keep.append(current.item())  # 将当前索引添加到保留列表中
        if indices.numel() == 1:
            break

        # 计算 IoU
        current_box = boxes[current].unsqueeze(0)
        boxes_left = boxes[indices[1:]]
        max_xy = torch.min(current_box[:, 2:], boxes_left[:, 2:])
        min_xy = torch.max(current_box[:, :2], boxes_left[:, :2])
        inter_wh = torch.clamp(max_xy - min_xy, min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        current_area = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        boxes_left_area = (boxes_left[:, 2] - boxes_left[:, 0]) * (boxes_left[:, 3] - boxes_left[:, 1])
        union_area = current_area + boxes_left_area - inter_area
        iou = inter_area / union_area

        # 保留 IoU 小于阈值的框
        indices = indices[(iou < iou_threshold).nonzero(as_tuple=True)[0] + 1]  # 修正索引

    return torch.tensor(keep)


if __name__ == '__main__':
    # 示例使用
    boxes = torch.tensor([[100, 100, 210, 210], [105, 105, 215, 215], [300, 300, 400, 400]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.75, 0.6], dtype=torch.float32)
    iou_threshold = 0.5

    from torchvision.ops import nms as tnms

    assert tnms(boxes, scores, iou_threshold) == nms(boxes, scores, iou_threshold)
