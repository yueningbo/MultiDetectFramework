import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import compute_iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=2, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per cell
        self.C = C  # Number of classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        target = self.convert_to_yolo_format(target, pred.device)
        return self.compute_loss(pred, target)

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
        N = 5 * B + C  # 5 = [x, y, w, h, conf]

        batch_size = pred.size(0)
        device = pred.device

        # Masks
        coord_mask = target[..., 4] > 0  # [batch_size, S, S]
        noobj_mask = target[..., 4] == 0  # Cells not containing objects
        # print('coord_mask:', coord_mask.shape)
        # print('noobj_mask:', noobj_mask.shape)

        # Expand masks
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)
        # print('Expanded coord_mask:', coord_mask.shape)
        # print('Expanded noobj_mask:', noobj_mask.shape)

        # Separate predictions and targets
        coord_pred = pred[coord_mask].reshape(-1, N)
        # print('coord_pred:', coord_pred.shape)
        # print('coord_pred[..., :5 * B]:', coord_pred[..., :5 * B].shape)

        bbox_pred = coord_pred[..., :5 * B].contiguous().view(-1, 5)
        class_pred = coord_pred[..., 5 * B:]
        # print('coord_pred:', coord_pred.shape)
        # print('bbox_pred:', bbox_pred.shape)
        # print('class_pred:', class_pred.shape)

        coord_target = target[coord_mask].view(-1, N)
        bbox_target = coord_target[..., :5 * B].contiguous().view(-1, 5)
        class_target = coord_target[..., 5 * B:]
        # print('coord_target:', coord_target.shape)
        # print('bbox_target:', bbox_target.shape)
        # print('class_target:', class_target.shape)

        # No object confidence loss
        noobj_pred = pred[noobj_mask].view(-1, N)
        noobj_target = target[noobj_mask].view(-1, N)
        # print('noobj_pred:', noobj_pred.shape)
        # print('noobj_target:', noobj_target.shape)

        noobj_conf_mask = torch.zeros(noobj_pred.size(), dtype=torch.bool, device=device)
        for b in range(B):
            noobj_conf_mask[:, 4 + b * 5] = 1
        # print('noobj_conf_mask:', noobj_conf_mask.shape)

        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        noobj_target_conf = noobj_target[noobj_conf_mask]
        # print('noobj_pred_conf:', noobj_pred_conf.shape)
        # print('noobj_target_conf:', noobj_target_conf.shape)

        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        # print('loss_noobj:', loss_noobj.item())

        # Object loss
        coord_response_mask = torch.zeros(bbox_target.size(), dtype=torch.bool, device=device)
        bbox_target_iou = torch.zeros(bbox_target.size(), device=device)
        # print('coord_response_mask:', coord_response_mask.shape)
        # print('bbox_target_iou:', bbox_target_iou.shape)

        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i + B]
            pred_xyxy = torch.zeros_like(pred)
            pred_xyxy[:, :2] = pred[:, :2] / S - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2] / S + 0.5 * pred[:, 2:4]

            target_bbox = bbox_target[i].view(-1, 5)
            target_xyxy = torch.zeros_like(target_bbox)
            target_xyxy[:, :2] = target_bbox[:, :2] / S - 0.5 * target_bbox[:, 2:4]
            target_xyxy[:, 2:4] = target_bbox[:, :2] / S + 0.5 * target_bbox[:, 2:4]

            iou = compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])
            max_iou, max_index = iou.max(0)
            max_index = max_index.item()

            coord_response_mask[i + max_index] = 1
            bbox_target_iou[i + max_index, 4] = max_iou.item()
            # print(f'Cell {i // B}: max_iou={max_iou.item()}, max_index={max_index}')

        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)
        # print('bbox_pred_response:', bbox_pred_response.shape)
        # print('bbox_target_response:', bbox_target_response.shape)
        # print('target_iou:', target_iou.shape)

        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]),
                             reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')
        # print('loss_xy:', loss_xy.item())
        # print('loss_wh:', loss_wh.item())
        # print('loss_obj:', loss_obj.item())

        # Class probability loss
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')
        # print('loss_class:', loss_class.item())

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / batch_size
        # print('self.lambda_coord * (loss_xy + loss_wh):', self.lambda_coord * (loss_xy + loss_wh).item())
        # print('self.lambda_noobj * loss_noobj:', self.lambda_noobj * loss_noobj.item())
        # print('loss_obj:', loss_obj.item())
        # print('loss_class:', loss_class.item())

        # print('total_loss:', loss.item())

        return loss

    def convert_to_yolo_format(self, targets, device):
        S, B, C = self.S, self.B, self.C
        batch_size = len(targets)
        converted_target = torch.zeros((batch_size, S, S, B * 5 + C), device=device)

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

                for b in range(B):
                    converted_target[bi, cell_y, cell_x, b * 5: b * 5 + 5] = torch.tensor(
                        [x_center_cell, y_center_cell, width, height, 1])

                converted_target[bi, cell_y, cell_x, B * 5 + label] = 1

        return converted_target


def test_yolov1_loss():
    S = 7
    B = 2
    C = 20
    batch_size = 1

    criterion = YOLOv1Loss(S=S, B=B, C=C)

    # 测试用例1：预测与目标完全匹配
    pred1 = torch.zeros((batch_size, S, S, B * 5 + C))
    target1 = torch.zeros((batch_size, S, S, B * 5 + C))
    target1[0, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0])
    target1[0, 3, 3, 5 * B + 5] = 1
    pred1[0, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0])
    pred1[0, 3, 3, 5 * B + 5] = 1
    loss1 = criterion.compute_loss(pred1, target1)

    expected_loss1 = 0.0
    assert abs(loss1.item() - expected_loss1) < 1e-6, f"Expected loss {expected_loss1}, but got {loss1.item()}"


if __name__ == "__main__":
    test_yolov1_loss()
