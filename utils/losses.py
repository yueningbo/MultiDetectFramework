from utils.utils import compute_iou
import torch
import torch.nn as nn


class YoloV1Loss(nn.Module):
    def __init__(self, config, lambda_coord=5, lambda_noobj=0.5, device=None):
        super(YoloV1Loss, self).__init__()
        self.S = config['grid_size']  # Grid size
        self.B = config['num_bounding_boxes']  # Number of bounding boxes per grid cell
        self.C = config['num_classes']  # Number of classes
        self.lambda_coord = lambda_coord  # Weight for the localization loss
        self.lambda_noobj = lambda_noobj  # Weight for the no-object confidence loss
        self.device = device

    def compute_localization_loss(self, pred_box, gt_box):
        """Compute localization loss (x_min, y_min, w, h)"""
        loc_loss = (pred_box[0] - gt_box[0]) ** 2 + (pred_box[1] - gt_box[1]) ** 2
        loc_loss += (torch.sqrt(pred_box[2]) - torch.sqrt(gt_box[2])) ** 2
        loc_loss += (torch.sqrt(pred_box[3]) - torch.sqrt(gt_box[3])) ** 2
        loc_loss = self.lambda_coord * loc_loss
        return loc_loss

    def compute_confidence_loss(self, pred_conf, iou, has_object):
        """Compute confidence loss"""
        if has_object:
            return (pred_conf - iou) ** 2
        else:
            return self.lambda_noobj * (pred_conf ** 2)

    def compute_classification_loss(self, pred_classes, gt_classes):
        """Compute classification loss"""
        return nn.CrossEntropyLoss()(pred_classes, gt_classes)

    def convert_to_one_hot(self, gt_classes, num_classes):
        one_hot = torch.zeros(num_classes, device=self.device)
        one_hot.scatter_(0, gt_classes.view(-1) - 1, 1)  # 这里考虑了background
        return one_hot

    def forward(self, outputs, targets):
        """
        outputs.shape: [batch, S, S, B*5 + C]
        targets: [{
            'boxes': tensor([[X_min,Y_min,W,H],...]),
            'labels': tensor([class_label,...])
            },...]
        """
        batch_size = outputs.size(0)
        total_loss = 0

        for i in range(batch_size):
            target = targets[i]
            output = outputs[i]

            gt_boxes = target['boxes']
            gt_classes = target['labels']
            one_hot_gt_classes = self.convert_to_one_hot(gt_classes, num_classes=self.C)

            for cell in range(self.S * self.S):
                row = cell // self.S
                col = cell % self.S

                cell_outputs = output[row, col, :]
                pred_boxes = cell_outputs[:self.B * 5].view(self.B, 5)
                pred_classes = cell_outputs[self.B * 5:]

                ious = compute_iou(pred_boxes[:, :4], gt_boxes)
                best_iou, best_box_idx = torch.max(ious, dim=1)

                for b in range(self.B):
                    if best_iou[b] > 0:
                        gt_box_idx = best_box_idx[b]
                        gt_box = gt_boxes[gt_box_idx]
                        pred_box = pred_boxes[b]

                        x_min = pred_box[0] + col / self.S
                        y_min = pred_box[1] + row / self.S
                        w = pred_box[2]
                        h = pred_box[3]

                        pred_box_adjusted = torch.tensor([x_min, y_min, w, h], device=self.device)

                        pred_conf = pred_box[4]

                        loc_loss = self.compute_localization_loss(pred_box_adjusted, gt_box)
                        conf_loss = self.compute_confidence_loss(pred_conf, best_iou[b], True)
                        class_loss = self.compute_classification_loss(pred_classes, one_hot_gt_classes)

                        total_loss += loc_loss + conf_loss + class_loss
                    else:
                        no_obj_conf_loss = self.compute_confidence_loss(pred_boxes[b, 4], 0, False)
                        total_loss += no_obj_conf_loss

        return total_loss / batch_size
