import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import compute_iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, C=2, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S  # Grid size
        self.C = C  # Number of classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        target = self.convert_to_yolo_format(target, pred.device)
        return self.loss_compute(pred, target)

    def compute_coord_loss(self, bbox_pred, bbox_target):
        coord_loss = F.mse_loss(bbox_pred[..., :2], bbox_target[..., :2], reduction='sum') + \
                     F.mse_loss(torch.sqrt(bbox_pred[..., 2:4] + 1e-6),
                                torch.sqrt(bbox_target[..., 2:4] + 1e-6),
                                reduction='sum')
        return self.lambda_coord * coord_loss

    def compute_conf_loss(self, conf_pred, conf_target, noobj_conf_pred, noobj_conf_target):
        conf_loss_obj = F.mse_loss(conf_pred, conf_target, reduction='sum')
        conf_loss_noobj = F.mse_loss(noobj_conf_pred, noobj_conf_target, reduction='sum')
        conf_loss = conf_loss_obj + self.lambda_noobj * conf_loss_noobj
        return conf_loss

    def compute_class_loss(self, class_pred, class_target):
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')
        return class_loss

    def loss_compute(self, pred, target):
        """ Compute loss for YOLO training.
        Args:
            pred: (Tensor) predictions, sized [batch_size, S, S, 5+C], 5=len([x, y, w, h, conf]).
            target: (Tensor) targets, sized [batch_size, S, S, 5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        S, C = self.S, self.C
        N = 5 + C  # 5 = [x, y, w, h, conf]

        batch_size = pred.size(0)

        # 区分正负样本
        coord_mask = target[..., 4] > 0  # [batch_size, S, S]
        noobj_mask = target[..., 4] == 0  # Cells not containing objects

        # Expand masks
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)  # [batch_size, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)  # [batch_size, S, S, N]

        # Separate predictions and targets
        coord_pred = pred[coord_mask].view(-1, N)  # [batch_size*S*S, N]
        bbox_pred = coord_pred[..., :5].contiguous().view(-1, 5)  # [batch_size*S*S, 5]
        class_pred = coord_pred[..., 5:]  # [batch_size*S*S, C]

        coord_target = target[coord_mask].view(-1, N)  # [batch_size*S*S, N]
        bbox_target = coord_target[..., :5].contiguous().view(-1, 5)  # [batch_size*S*S, 5]
        class_target = coord_target[..., 5:]  # [batch_size*S*S, C]

        conf_pred = bbox_pred[..., 4]  # [N,]
        conf_target = torch.zeros_like(conf_pred)  # [N]
        # 计算bbox_pred的最大iou
        for i in range(bbox_pred.size(0)):
            temp_bbox_pred = bbox_pred[i].expand_as(bbox_target)  # [N,]
            iou_pred = compute_iou(temp_bbox_pred, bbox_target)  # [N,]
            conf_target[i] = iou_pred.max()

        # No object confidence loss
        noobj_pred = pred[noobj_mask].view(-1, N)  # [batch_size*S*S, N]
        noobj_target = target[noobj_mask].view(-1, N)  # [batch_size*S*S, N]

        noobj_conf_pred = noobj_pred[..., 4]
        noobj_conf_target = noobj_target[..., 4]

        # Compute losses
        coord_loss = self.compute_coord_loss(bbox_pred, bbox_target)
        conf_loss = self.compute_conf_loss(conf_pred, conf_target, noobj_conf_pred, noobj_conf_target)
        class_loss = self.compute_class_loss(class_pred, class_target)

        logging.debug(f'coord_loss:{coord_loss.item()}')
        logging.debug(f'conf_loss:{conf_loss.item()}')
        logging.debug(f'class_loss:{class_loss.item()}')
        logging.debug(f'---------------------------------')

        # Total loss
        total_loss = coord_loss + conf_loss + class_loss

        return total_loss / batch_size

    def convert_to_yolo_format(self, targets, device):
        S, C = self.S, self.C
        batch_size = len(targets)
        converted_target = torch.zeros((batch_size, S, S, 5 + C), device=device)

        for bi in range(batch_size):
            target = targets[bi]
            boxes = target['boxes']
            labels = target['labels']
            for box, label in zip(boxes, labels):
                x_min, y_min, width, height = box
                cell_x = int(x_min * S)
                cell_y = int(y_min * S)
                x_center_cell = x_min * S - cell_x
                y_center_cell = y_min * S - cell_y

                converted_target[bi, cell_y, cell_x, :5] = torch.tensor(
                    [x_center_cell, y_center_cell, width, height, 1])

                converted_target[bi, cell_y, cell_x, 5 + label] = 1

        return converted_target


def test_yolov1_loss():
    S = 7
    C = 20
    batch_size = 1

    criterion = YOLOv1Loss(S=S, C=C)

    # 测试用例1：预测与目标完全匹配
    pred1 = torch.zeros((batch_size, S, S, 5 + C))
    target1 = torch.zeros((batch_size, S, S, 5 + C))
    target1[0, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0])
    target1[0, 3, 3, 5 + 5] = 1
    pred1[0, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0])
    pred1[0, 3, 3, 5 + 5] = 1
    loss1 = criterion.loss_compute(pred1, target1)

    expected_loss1 = 0.0
    assert abs(loss1.item() - expected_loss1) < 1e-6, f"Expected loss {expected_loss1}, but got {loss1.item()}"


if __name__ == "__main__":
    test_yolov1_loss()
