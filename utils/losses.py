import torch
import torch.nn.functional as F

from utils.utils import bbox_iou


def compute_loss(outputs, targets, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
    """
    Compute the YOLOv1 loss.
    """
    # Extract the different components of the outputs
    pred_boxes = outputs[..., :4]  # Bounding box coordinates
    pred_conf = outputs[..., 4:5]  # Objectness score
    pred_cls = outputs[..., 5:]  # Class scores

    # Extract true targets
    true_boxes = torch.stack([t['boxes'] for t in targets], dim=0)  # Stack boxes into a tensor
    true_cls = torch.stack([t['labels'] for t in targets], dim=0)  # Stack labels into a tensor
    true_conf = (true_boxes > 0).float().sum(dim=1,
                                             keepdim=True)  # Assuming non-zero boxes indicate presence of objects

    # Prepare target tensors
    true_boxes = true_boxes.view(-1, 4)  # Flatten for IoU calculation
    true_cls = true_cls.view(-1)
    true_conf = true_conf.view(-1)

    # Calculate the bounding box loss
    iou = bbox_iou(pred_boxes.view(-1, 4), true_boxes)
    bbox_loss = 1 - iou.max(dim=1)[0]  # IoU loss between the predicted and true boxes

    # Calculate the confidence loss
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf.view(-1, 1), true_conf.view(-1, 1), reduction='none')

    # Calculate the class loss
    cls_loss = F.cross_entropy(pred_cls.view(-1, C), true_cls.view(-1), reduction='none')

    # Combine the losses
    total_loss = (lambda_coord * bbox_loss + conf_loss + lambda_noobj * (1 - true_conf) * conf_loss + cls_loss).mean()

    return total_loss
