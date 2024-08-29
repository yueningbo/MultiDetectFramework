from utils.utils import compute_iou
import torch
import torch.nn as nn


class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5, device=None):
        super(YoloV1Loss, self).__init__()
        self.S = S  # 网格大小
        self.B = B  # 每个网格预测的边界框数量
        self.C = C  # 类别数量
        self.lambda_coord = lambda_coord  # 定位损失的权重
        self.lambda_noobj = lambda_noobj  # 没有物体时置信度损失的权重
        self.device = device

    def compute_localization_loss(self, pred_box, gt_box):
        """计算定位损失 (x, y, w, h)"""
        loc_loss = (pred_box[0] - gt_box[0]) ** 2 + (pred_box[1] - gt_box[1]) ** 2
        loc_loss += (torch.sqrt(pred_box[2]) - torch.sqrt(gt_box[2])) ** 2
        loc_loss += (torch.sqrt(pred_box[3]) - torch.sqrt(gt_box[3])) ** 2
        loc_loss = self.lambda_coord * loc_loss
        return loc_loss

    def compute_confidence_loss(self, pred_conf, iou, has_object):
        """计算置信度损失"""
        if has_object:
            return (pred_conf - iou) ** 2
        else:
            return self.lambda_noobj * (pred_conf ** 2)

    def compute_classification_loss(self, pred_classes, gt_classes):
        """计算分类损失"""
        return nn.CrossEntropyLoss()(pred_classes, gt_classes)

    def convert_to_one_hot(self, gt_classes, num_classes):
        one_hot = torch.zeros(num_classes).to(device=self.device)
        one_hot.scatter_(0, gt_classes.view(-1), 1)
        return one_hot

    def forward(self, outputs, targets):
        """
        outputs.shape: [batch, H, W, C]
        targets: [{
            'boxes': tensor([[X,Y,H,W],...]),
            'label':tensor([class_label,...])
            },...]
        """
        batch_size = outputs.size(0)
        total_loss = 0

        for i in range(batch_size):
            target = targets[i]
            output = outputs[i]

            # 提取目标框和类别
            gt_boxes = target['boxes']
            gt_classes = target['labels']  # tensor([2], device='cuda:0')
            one_hot_gt_classes = self.convert_to_one_hot(gt_classes, num_classes=self.C)

            for cell in range(self.S * self.S):
                row = cell // self.S
                col = cell % self.S

                # 分割预测输出
                cell_outputs = output[row, col, :]
                pred_boxes = cell_outputs[:self.B * 5].view(self.B, 5)
                pred_classes = cell_outputs[self.B * 5:]  # torch.Size([20])

                # 找出与每个预测框重叠度最高的真实框
                best_iou = torch.zeros(self.B).to(self.device)
                best_box_idx = -torch.ones(self.B, dtype=torch.int8).to(self.device)

                # Compute IoU for each predicted box
                for b in range(self.B):
                    pred_box = pred_boxes[b, :4]
                    ious = compute_iou(pred_box.unsqueeze(0), gt_boxes)
                    best_iou[b], best_box_idx[b] = torch.max(ious, dim=0)

                for b in range(self.B):
                    if best_iou[b] > 0:  # 存在匹配的真实框
                        gt_box_idx = best_box_idx[b]
                        gt_box = gt_boxes[gt_box_idx]
                        pred_box = pred_boxes[b]
                        # 预测的置信度
                        pred_conf = pred_box[4]

                        # 计算各个损失
                        loc_loss = self.compute_localization_loss(pred_box, gt_box)
                        conf_loss = self.compute_confidence_loss(pred_conf, best_iou[b], True)
                        class_loss = self.compute_classification_loss(pred_classes, one_hot_gt_classes)

                        total_loss += loc_loss + conf_loss + class_loss
                    else:  # 该网格不包含目标
                        no_obj_conf_loss = self.compute_confidence_loss(pred_boxes[b, 4], 0, False)
                        total_loss += no_obj_conf_loss

        return total_loss / batch_size  # 返回平均损失
