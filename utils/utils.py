import torch


def bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x1g, y1g, x2g, y2g = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter = (torch.min(x2, x2g) - torch.max(x1, x1g)) * (torch.min(y2, y2g) - torch.max(y1, y1g)).clamp(min=0)
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter
    IoU = inter / union
    return IoU
