import logging
import os

import torch
import torch.nn as nn

from models.yolov1.backbone import Darknet
from utils.utils import center_to_corners, nms


class YOLOv1(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, num_bounding_boxes=2, img_orig_size=448,
                 pretrained_weights_path=None):
        super().__init__()

        self.S = grid_size  # Grid size
        self.B = num_bounding_boxes  # Number of bounding boxes
        self.C = num_classes  # Number of classes
        self.img_orig_size = img_orig_size  # Original image size

        self.darknet = Darknet()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 7 * 7 * (num_classes + 5 * self.B)),
            nn.Sigmoid()
        )

        if pretrained_weights_path:
            self._load_pretrained_weights(pretrained_weights_path)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.darknet(x)
        x = self.fc(x)
        x = x.view(-1, 7, 7, 5 * self.B + self.C)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self, path):
        if os.path.isfile(path):
            logging.info(f"Loading weights from {path}")
            state_dict = torch.load(path)
            self.load_state_dict(state_dict, strict=False)
        else:
            logging.error(f"Pretrained weights file not found at {path}. Using initialized weights.")

    def decode_boxes(self, bboxes, grid_size, img_size):
        """
        Decode the predicted bounding boxes from the grid cell coordinates to image coordinates.

        Args:
            bboxes (Tensor): Predicted bounding boxes in grid cell coordinates, shape [S, S, B, 4]
            grid_size (int): The size of the grid (S)
            img_size (int): The original image size

        Returns:
            Tensor: Decoded bounding boxes in image coordinates, shape [S, S, B, 4]
        """
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([grid_size, grid_size, 1]).to(bboxes.device)
        grid_y = grid_x.permute(1, 0, 2)

        bx = (bboxes[..., 0] + grid_x) / grid_size * img_size
        by = (bboxes[..., 1] + grid_y) / grid_size * img_size
        bw = bboxes[..., 2] * img_size
        bh = bboxes[..., 3] * img_size

        decoded_bboxes = torch.stack([bx, by, bw, bh], dim=-1)
        return decoded_bboxes

    def post_process(self, bboxes, scores, conf_thresh, nms_thresh):
        """
        根据得分获取预测的类别标签，然后进行阈值筛选，再按类别进行非极大值抑制。

        Args:
            bboxes (Tensor): 形状为[H, W, B, 5]
            scores (Tensor): 形状为[H, W, C]
            conf_thresh (float): 置信度阈值
            nms_thresh (float): 非极大值抑制阈值

        Returns:
            tuple: bboxes, scores, labels
        """
        bboxes = self.decode_boxes(bboxes, self.S, self.img_orig_size)
        bboxes = bboxes.view(-1, self.B, 4)
        scores = scores.view(-1, self.C)

        max_scores, _ = scores.max(dim=-1)
        conf_mask = max_scores > conf_thresh
        scores = scores[conf_mask]
        bboxes = bboxes[conf_mask].view(-1, 4)
        bboxes = center_to_corners(bboxes)

        all_bboxes, all_scores, all_labels = [], [], []
        for class_idx in range(self.C):
            class_scores = scores[:, class_idx]
            class_bboxes = bboxes

            keep = nms(class_bboxes, class_scores, nms_thresh)
            all_bboxes.append(class_bboxes[keep])
            all_scores.append(class_scores[keep])
            all_labels.extend([class_idx + 1] * len(keep))

        if len(all_bboxes) > 0:
            all_bboxes = torch.cat(all_bboxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
        else:
            all_bboxes = torch.empty((0, 4), device=next(self.parameters()).device)
            all_scores = torch.empty((0,), device=next(self.parameters()).device)

        return all_bboxes, all_scores, all_labels

    @torch.no_grad()
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.4):
        """
        执行模型推理过程。

        Args:
            images (Tensor): 输入图像张量，形状为 [batch, H, W, C]
            conf_thresh (float): 置信度阈值
            nms_thresh (float): 非极大值抑制阈值

        Returns:
            list: [(bboxes, scores, labels),...]
        """
        outputs = self(images)
        bboxes, scores, labels = [], [], []
        for i in range(images.size(0)):
            output = outputs[i]
            bboxes_i = output[..., :self.B * 5].view(self.S, self.S, self.B, 5)  # 形状为[H, W, B, 5]
            scores_i = output[..., self.B * 5:]  # 形状为[H, W, C]
            bboxes_i, scores_i, labels_i = self.post_process(bboxes_i, scores_i, conf_thresh, nms_thresh)
            bboxes.append(bboxes_i)
            scores.append(scores_i)
            labels.append(labels_i)

        return list(zip(bboxes, scores, labels))
