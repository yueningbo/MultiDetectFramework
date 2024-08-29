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
            nn.Conv2d(512, (self.C + self.B * 5), kernel_size=1)
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
                  where each inner list is [[x_min, y_min, width, height, class_id, confidence]].
        """
        grid_size = output.size(0)
        processed_results = []

        output = output.view(grid_size * grid_size, -1)  # [grid_size*grid_size, C]

        # Confidence filtering
        conf_mask = output[:, 4] > conf_threshold
        output = output[conf_mask]

        # Parse bounding boxes

        # Perform NMS
        final_boxes = []
        for class_id in range(self.C):
            class_mask = output[:, self.B * 5 + class_id] > conf_threshold

            # Apply mask to output, filtering only the relevant rows
            filtered_output = output[class_mask]

            if filtered_output.shape[0] == 0:  # If no boxes pass the threshold, skip to the next class
                continue

            # Extract bounding boxes for the class
            class_boxes = filtered_output[:, :self.B * 5].reshape(-1, 5)
            class_scores = filtered_output[:, self.B * 5 + class_id]

            corners_class_boxes = center_to_corners(class_boxes)
            keep = nms(corners_class_boxes, class_scores, nms_threshold)

            for idx in keep:
                box = class_boxes[idx]
                box *= self.img_orig_size
                # class_id+1是因为background占了1个位置
                final_boxes.append([box[0], box[1], box[2], box[3], class_id + 1, class_scores[idx].item()])

        processed_results.extend(final_boxes)

        return processed_results

    @torch.no_grad()
    def inference(self, images, conf_threshold=0.5, nms_threshold=0.4):
        """
        执行模型推理过程

        Args:
            images (Tensor): 输入图像张量，形状为 [batch_size, C, H, W]
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
