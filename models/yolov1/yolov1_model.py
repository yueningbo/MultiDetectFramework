import logging
import os

import torch
import torch.nn as nn
import torchvision.models as models

from utils.utils import center_to_corners, nms


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20, pretrained_weights_path=None, img_orig_size=448):
        super().__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes
        self.C = C  # Number of classes
        self.img_orig_size = img_orig_size

        # Load pretrained MobileNetV2 as the backbone
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        # Use MobileNetV2 up to the penultimate layer
        self.backbone = nn.Sequential(
            *list(mobilenet_v2.features.children()),  # Keep all layers
            nn.AdaptiveAvgPool2d((self.S, self.S))  # Adjust to SxS grid
        )

        # Reduce the number of output channels to make it simpler
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),  # Reducing output channels
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, (self.B * 5 + self.C), kernel_size=1),
        )

        if pretrained_weights_path:
            self._load_pretrained_weights(pretrained_weights_path)
        else:
            logging.info('Model weights init.')
            self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_layers(x)

        # [batch, C, H, W] -> [batch, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize weights using He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm layers
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
        # Grid cell offsets, shape: [S, S, 1]
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([grid_size, grid_size, 1]).to(
            bboxes.device)
        grid_y = grid_x.permute(1, 0, 2)

        # Decode bbox center
        bx = (bboxes[..., 0] + grid_x) / grid_size * img_size
        by = (bboxes[..., 1] + grid_y) / grid_size * img_size

        # Decode bbox width and height (assuming they are predicted as sqrt(w) and sqrt(h))
        bw = bboxes[..., 2] * img_size
        bh = bboxes[..., 3] * img_size

        decoded_bboxes = torch.stack([bx, by, bw, bh], dim=-1)
        return decoded_bboxes

    def post_process(self, bboxes, scores, conf_thresh, nms_thresh):
        """
        根据得分获取预测的类别标签
        然后进行阈值筛选
        再按类别进行非极大值抑制
        :param bboxes: 形状为[S, S, B, 4]
        :param scores: 形状为[S, S, C]
        :param conf_thresh: 置信度阈值
        :param nms_thresh: 非极大值抑制阈值
        :return: bboxes, scores, labels
        """
        # Decode boxes from grid cell to image coordinates
        bboxes = self.decode_boxes(bboxes, self.S, self.img_orig_size)

        bboxes = bboxes.view(-1, self.B, 4)  # Flatten to [N, B, 4]
        scores = scores.view(-1, self.C)  # Flatten to [N, C]

        max_scores, _ = scores.max(dim=-1)
        conf_mask = max_scores > conf_thresh
        scores = scores[conf_mask]
        bboxes = bboxes[conf_mask]

        bboxes = bboxes.view(-1, 4)

        # Convert center format to corner format
        bboxes = center_to_corners(bboxes)

        all_bboxes, all_scores, all_labels = [], [], []
        for class_idx in range(self.C):
            class_scores = scores[:, class_idx]
            class_bboxes = bboxes

            # Apply NMS per class
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
        执行模型推理过程

        Args:
            images (Tensor): 输入图像张量，形状为 [batch, H, W, C]
            conf_threshold (float): 置信度阈值
            nms_threshold (float): 非极大值抑制阈值

        Returns:
            list: [(bboxes, scores, labels),...]
        """
        outputs = self(images)
        bboxes, scores, labels = [], [], []
        for i in range(images.size(0)):
            output = outputs[i]  # [H, W, (B*5 + C)]
            bboxes_i = output[..., :self.B * 4].view(self.S, self.S, self.B, 4)
            scores_i = output[..., self.B * 5:]
            bboxes_i, scores_i, labels_i = self.post_process(bboxes_i, scores_i, conf_thresh, nms_thresh)
            bboxes.append(bboxes_i)
            scores.append(scores_i)
            labels.append(labels_i)

        return list(zip(bboxes, scores, labels))
