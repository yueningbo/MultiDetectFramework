import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import compute_iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=3, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per cell
        self.C = C  # Number of classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        converted_target = self.convert_to_yolo_format(target)
        self.loss_compute(pred, converted_target)

    def loss_compute(self, pred, target):
        """ Compute loss for YOLO training.
        Args:
            pred: (Tensor) predictions, sized [batch_size, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target: (Tensor) targets, sized [batch_size, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        # TODO: Romove redundant dimensions for some Tensors.

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C  # 5=len([x, y, w, h, conf])

        device = pred.device

        batch_size = pred.size(0)

        coord_mask = target[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target[:, :, :, 4] == 0  # mask for the cells which do not contain objects. [n_batch, S, S]

        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)  # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)  # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = pred[coord_mask].view(-1, N)  # pred tensor on the cells which contain objects. [n_coord, N]
        # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)  # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_pred = coord_pred[:, 5 * B:]  # [n_coord, C]

        coord_target = target[coord_mask].view(-1, N)  # target tensor on the cells which contain objects. [n_coord, N]
        # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5 * B].contiguous().view(-1, 5)  # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = coord_target[:, 5 * B:]  # [n_coord, C]

        # Compute loss for the cells with no object bbox.
        noobj_pred = pred[noobj_mask].view(-1, N)  # pred on the cells which do not contain objects. [n_noobj, N]
        noobj_target = target[noobj_mask].view(-1, N)  # target on the cells which do not contain objects. [n_noobj, N]

        noobj_conf_mask = torch.zeros(noobj_pred.size(), dtype=torch.bool, device=device)  # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:, 4 + b * 5] = 1  # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.zeros(bbox_target.size(), dtype=torch.bool, device=device)  # [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size(),
                                      device=device)  # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i + B]  # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = torch.zeros_like(pred)  # [B, 5=len([x1, y1, x2, y2, conf])]
            pred_xyxy[:, :2] = pred[:, :2] / float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2] / float(S) + 0.5 * pred[:, 2:4]

            target_bbox = bbox_target[i].view(-1, 5)  # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = torch.zeros_like(target_bbox)  # [1, 5=len([x1, y1, x2, y2, conf])]
            target_xyxy[:, :2] = target_bbox[:, :2] / float(S) - 0.5 * target_bbox[:, 2:4]
            target_xyxy[:, 2:4] = target_bbox[:, :2] / float(S) + 0.5 * target_bbox[:, 2:4]

            iou = compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])  # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.item()

            coord_response_mask[i + max_index] = 1

            bbox_target_iou[i + max_index, 4] = max_iou.item()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)  # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1,
                                                                     5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1,
                                                               5)  # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]),
                             reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss

    def convert_to_yolo_format(self, targets):
        """
        将目标框和标签转换为 YOLO 损失函数所需的格式。

        Args:
            boxes (Tensor): 边界框，形状为 [N, 4]，其中 N 是边界框的数量。
            labels (Tensor): 标签，形状为 [N]。
        Returns:
            target (Tensor): 转换后的目标张量，形状为 [batch_size, S, S, B * 5 + C]。
        """
        S = self.S
        B = self.B
        C = self.C

        batch_size = len(targets)


        print((batch_size, S, S, B * 5 + C))
        converted_target = torch.zeros((batch_size, S, S, B * 5 + C))

        for bi in range(batch_size):
            boxes = targets['boxes']
            labels = targets['labels']
            for box, label in zip(boxes, labels):
                x_min, y_min, width, height = box

                cell_x = int(x_min * S)
                cell_y = int(y_min * S)
                x_center_cell = x_min * S - cell_x
                y_center_cell = y_min * S - cell_y

                for b in range(B):
                    converted_target[bi, cell_y, cell_x, b * 5: b * 5 + 5] = torch.tensor(
                        [x_center_cell, y_center_cell, width, height, 1])

                converted_target[bi, cell_y, cell_x, B * 5 + label] = 1

        return target


# 测试用例
if __name__ == "__main__":
    # 创建随机的预测张量和目标张量
    batch_size = 2
    S = 7  # Grid size
    B = 2  # Number of bounding boxes per cell
    C = 20  # Number of classes

    # 预测张量：大小为 [batch_size, S, S, B*5 + C]
    pred = torch.rand((batch_size, S, S, B * 5 + C))
    # 目标张量：大小为 [batch_size, S, S, B*5 + C]
    target = torch.rand((batch_size, S, S, B * 5 + C))

    # 实例化损失函数
    criterion = YOLOv1Loss(S=S, B=B, C=C)

    # 计算损失
    loss = criterion(pred, target)

    # 输出损失值
    print(f'Test loss: {loss.item()}')
