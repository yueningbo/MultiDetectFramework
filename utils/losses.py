import torch
import torch.nn.functional as F
from utils.utils import bbox_iou


def compute_loss(outputs, targets, lambda_coord=5, lambda_noobj=0.5):
    """
    Compute the loss for YOLOv1.

    Parameters:
    outputs (torch.Tensor): The model outputs with shape (batch_size, 7*7*30).
    targets (torch.Tensor): The ground truth targets with shape (batch_size, 7*7*(5+num_classes)).
    num_classes (int): The number of object classes.
    lambda_coord (float): The weight of the bounding box coordinate loss.
    lambda_noobj (float): The weight of the loss for objects not present in the cell.

    Returns:
    torch.Tensor: The computed loss.
    """
    # Extract the different components of the outputs and targets
    pred_boxes = outputs[..., :4]  # Bounding box coordinates
    pred_conf = outputs[..., 4:5]  # Objectness score
    pred_cls = outputs[..., 5:]  # Class scores

    true_boxes = targets[..., :4]
    true_conf = targets[..., 4:5]
    true_cls = targets[..., 5:]

    # Calculate the bounding box loss
    # IoU loss between the predicted and true boxes
    bbox_loss = 1 - torch.diag(bbox_iou(pred_boxes.unsqueeze(2), true_boxes.unsqueeze(1)))

    # Calculate the confidence loss
    # If the ground truth box is empty, the confidence should be 0; otherwise, it should be 1
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, true_conf, reduction='none')

    # Calculate the class loss
    # Use CrossEntropyLoss, ignoring the confidence dimension
    cls_loss = F.cross_entropy(pred_cls, true_cls.argmax(dim=2), reduction='none')

    # Combine the losses
    total_loss = (lambda_coord * bbox_loss + conf_loss + lambda_noobj * (1 - true_conf) * conf_loss + cls_loss).mean()

    return total_loss
