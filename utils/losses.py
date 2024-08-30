import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import compute_iou


class YoloV1Loss(nn.Module):
    def __init__(self, config, lambda_coord=5, lambda_noobj=0.5, device=None):
        super(YoloV1Loss, self).__init__()
        self.S = config['grid_size']  # Grid size
        self.B = config['num_bounding_boxes']  # Number of bounding boxes per grid cell
        self.C = config['num_classes']  # Number of classes
        self.lambda_coord = lambda_coord  # Weight for the localization loss
        self.lambda_noobj = lambda_noobj  # Weight for the no-object confidence loss
        self.device = device

    def forward(self, outputs, targets):
        """
        outputs.shape: [batch, S, S, B*5 + C]
        targets: [{
            'boxes': tensor([[x_center, y_center, width, height],...]),  # Normalized by image size
            'labels': tensor([class_label,...])
        },...]
        """
        N = outputs.size(0)
        total_loss = 0

        for i in range(N):
            output = outputs[i]  # [S, S, B*5 + C]
            target = targets[i]

            # Reshape output to [S, S, B, 5 + C] to separate bbox and class predictions
            output = output.view(self.S, self.S, self.B, 5 + self.C)

            # Extract elements
            pred_boxes = output[..., :self.B * 4].contiguous().view(self.S, self.S, self.B, 4)  # [S, S, B, 4]
            pred_conf = output[..., self.B * 4:self.B * 5].squeeze(-1)  # [S, S, B]
            pred_cls = output[..., self.B * 5:]  # [S, S, C]

            # Convert predicted box coordinates to (x, y, w, h) format
            grid_x = torch.arange(self.S).repeat(self.S, 1).view(self.S, self.S, 1).to(self.device)
            grid_y = grid_x.permute(1, 0, 2)
            cell_size = 1.0 / self.S

            pred_boxes[..., 0] = (pred_boxes[..., 0] + grid_x) * cell_size  # x_center
            pred_boxes[..., 1] = (pred_boxes[..., 1] + grid_y) * cell_size  # y_center
            pred_boxes[..., 2] = pred_boxes[..., 2]  # width
            pred_boxes[..., 3] = pred_boxes[..., 3]  # height

            # Prepare target data
            target_boxes = target['boxes']  # [num_boxes, 4]
            target_labels = target['labels']  # [num_boxes]
            # 将标签转换为one-hot格式
            target_labels_one_hot = F.one_hot(target_labels, num_classes=self.C).float()  # [num_boxes, C]

            box_loss = 0
            class_loss = 0
            noobj_loss = 0
            obj_loss = 0

            # Create a mask for identifying cells with objects
            obj_mask = torch.zeros(self.S, self.S, self.B, dtype=torch.bool).to(self.device)

            for target_box, target_label_one_hot in zip(target_boxes, target_labels_one_hot):
                tx, ty, tw, th = target_box

                # Convert box center (cx, cy) to grid coordinates
                grid_x_idx = int(tx / cell_size)
                grid_y_idx = int(ty / cell_size)

                # Calculate offset within the grid cell
                offset_x = (tx - grid_x_idx * cell_size) / cell_size
                offset_y = (ty - grid_y_idx * cell_size) / cell_size

                # Find the best bounding box predictor (one with highest IoU)
                best_iou = 0
                best_box = None
                best_index = 0

                for b in range(self.B):
                    pred_box = pred_boxes[grid_y_idx, grid_x_idx, b]
                    pred_x, pred_y, pred_w, pred_h = pred_box

                    iou = compute_iou(
                        torch.tensor([[tx, ty, tw, th]]).to(self.device),
                        torch.tensor([[pred_x, pred_y, pred_w, pred_h]]).to(self.device)
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_box = pred_box
                        best_index = b

                # Calculate box loss using the best predictor
                pred_x, pred_y, pred_w, pred_h = best_box
                box_loss += F.mse_loss(pred_x, offset_x) + F.mse_loss(pred_y, offset_y)

                # 使用clamp来避免数值不稳定性
                pred_w = torch.clamp(pred_w, min=1e-6)
                pred_h = torch.clamp(pred_h, min=1e-6)
                tw = torch.clamp(tw, min=1e-6)
                th = torch.clamp(th, min=1e-6)

                box_loss += F.mse_loss(torch.sqrt(pred_w), torch.sqrt(tw)) + F.mse_loss(torch.sqrt(pred_h),
                                                                                        torch.sqrt(th))

                # Calculate confidence loss for the best predictor
                obj_loss += F.mse_loss(pred_conf[grid_y_idx, grid_x_idx, best_index],
                                       torch.tensor([1.0]).to(self.device))

                # Calculate classification loss
                class_loss += F.softmax(pred_cls[grid_y_idx, grid_x_idx], target_label_one_hot)

            # Calculate no-object confidence loss for other bounding boxes
            # Calculate no-object confidence loss for bounding boxes without objects
            noobj_loss += F.mse_loss(pred_conf[~obj_mask], torch.zeros_like(pred_conf[~obj_mask]))

            total_loss += self.lambda_coord * box_loss + class_loss + obj_loss + self.lambda_noobj * noobj_loss

        total_loss /= N
        return total_loss
