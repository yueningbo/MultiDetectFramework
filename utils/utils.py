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


if __name__ == '__main__':
    # 测试数据
    pred_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],  # box1: [x_center, y_center, width, height]
        [0.8, 0.8, 0.2, 0.2]
    ], dtype=torch.float32)

    gt_boxes = torch.tensor([
        [0.6, 0.6, 0.2, 0.2],
        [0.7, 0.7, 0.1, 0.1],
        [0.9, 0.9, 0.2, 0.2]
    ], dtype=torch.float32)

    # 计算 IoU
    ious = compute_iou(pred_boxes, gt_boxes)

    # 找到每个预测框与所有真实框之间的最大 IoU 和对应的索引
    best_iou, best_box_idx = torch.max(ious, dim=1)

    print("IoU Matrix:")
    print(ious)
    print("Best IoU for each predicted box:")
    print(best_iou)
    print("Index of the best matching ground truth box for each predicted box:")
    print(best_box_idx)


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
