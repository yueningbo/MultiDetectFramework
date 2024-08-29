from utils.utils import bbox_iou
import torch
import torch.nn as nn


class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloV1Loss, self).__init__()
        self.S = S  # 网格大小
        self.B = B  # 每个网格预测的边界框数量
        self.C = C  # 类别数量
        self.lambda_coord = lambda_coord  # 定位损失的权重
        self.lambda_noobj = lambda_noobj  # 没有物体时置信度损失的权重

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        total_loss = 0

        for i in range(batch_size):
            target = targets[i]
            output = outputs[i]

            for cell in range(self.S * self.S):
                row = cell // self.S
                col = cell % self.S

                # 分割预测输出
                cell_outputs = output[row, col, :]
                pred_boxes = cell_outputs[:self.B * 5].view(self.B, 5)
                pred_classes = cell_outputs[self.B * 5:]

                # 提取目标框和类别
                gt_boxes = target['boxes']
                gt_classes = target['labels']

                # 找出与每个预测框重叠度最高的真实框
                best_iou = 0
                best_box_idx = -1
                for j in range(len(gt_boxes)):
                    iou = bbox_iou(pred_boxes[:, :4], gt_boxes[j:j + 1])
                    if iou.max() > best_iou:
                        best_iou = iou.max()
                        best_box_idx = iou.argmax()

                if best_box_idx >= 0:  # 存在匹配的真实框
                    gt_box = gt_boxes[best_box_idx]

                    # 定位损失 (x, y, w, h)
                    pred_box = pred_boxes[best_box_idx]
                    loc_loss = (pred_box[0] - gt_box[0]) ** 2 + (pred_box[1] - gt_box[1]) ** 2
                    loc_loss += (torch.sqrt(pred_box[2]) - torch.sqrt(gt_box[2])) ** 2
                    loc_loss += (torch.sqrt(pred_box[3]) - torch.sqrt(gt_box[3])) ** 2
                    loc_loss = self.lambda_coord * loc_loss

                    # 置信度损失
                    conf_loss = (pred_box[4] - best_iou) ** 2

                    # 分类损失 (只在包含物体的单元格上计算)
                    class_loss = nn.CrossEntropyLoss()(pred_classes, gt_classes)

                    total_loss += loc_loss + conf_loss + class_loss
                else:  # 该网格不包含目标
                    no_obj_conf_loss = self.lambda_noobj * (pred_boxes[:, 4] ** 2).sum()
                    total_loss += no_obj_conf_loss

        return total_loss / batch_size  # 返回平均损失
