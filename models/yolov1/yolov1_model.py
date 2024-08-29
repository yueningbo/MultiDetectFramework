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
            nn.Conv2d(512, (self.B * 5 + self.C), kernel_size=1)
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

    def post_process(self, output: torch.Tensor, conf_threshold=0.5, nms_threshold=0.4):
        """
        Post-processes the model output, including confidence filtering and Non-Maximum Suppression (NMS).

        Args:
            output (Tensor): Model output of shape [H, W, C].
            conf_threshold (float): Confidence threshold.
            nms_threshold (float): NMS threshold.

        Returns:
            list: Processed detection results as a list of lists,
                  where each inner list is [x_min, y_min, width, height, class_id, confidence].
        """
        grid_size = output.size(0)
        processed_results = []

        for row in range(grid_size):
            for col in range(grid_size):
                cell_output = output[row, col, :]
                print(f'cell_output.shape:{cell_output.shape}')
                pred_boxes = cell_output[:self.B * 5].view(self.B, 5)  # [B, 5]
                pred_classes = cell_output[self.B * 5:]  # [C]

                # Compute class scores
                scores = pred_boxes[:, 4] * pred_classes  # Confidence score for each class

                # Filter boxes by class confidence threshold
                valid_mask = scores > conf_threshold
                pred_boxes = pred_boxes[valid_mask]
                class_ids = pred_classes[valid_mask]
                class_scores = scores[valid_mask]

                if len(pred_boxes) == 0:
                    continue

                # Adjust the bounding boxes based on cell position
                pred_boxes[:, 0] += col / grid_size
                pred_boxes[:, 1] += row / grid_size

                corners_class_boxes = center_to_corners(pred_boxes)
                keep = nms(corners_class_boxes, class_scores, nms_threshold)

                for idx in keep:
                    box = pred_boxes[idx]
                    # Convert coordinates to the original image size
                    box *= self.img_orig_size
                    # Append results (class_id + 1 for background class adjustment)
                    processed_results.append(
                        [box[0], box[1], box[2], box[3], class_ids[idx].item() + 1, class_scores[idx].item()])

        return processed_results

    @torch.no_grad()
    def inference(self, images, conf_threshold=0.5, nms_threshold=0.4):
        """
        执行模型推理过程

        Args:
            images (Tensor): 输入图像张量，形状为 [batch, H, W, C]
            conf_threshold (float): 置信度阈值
            nms_threshold (float): 非极大值抑制阈值

        Returns:
            list: 每张图像的检测结果列表
        """
        self.eval()
        images = images.to(next(self.parameters()).device)  # Ensure the data is on the correct device
        all_results = []

        outputs = self(images)
        for i in range(images.size(0)):
            output = outputs[i]
            processed_results = self.post_process(output, conf_threshold, nms_threshold)
            all_results.append(processed_results)

        return all_results
