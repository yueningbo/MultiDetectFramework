import logging
import os

import torch
import torch.nn as nn

from models.yolov1.backbone import Darknet
from utils.utils import nms, xywh_to_xyxy


class YOLOv1(nn.Module):
    def __init__(self, grid_size=7, num_bounding_boxes=2, num_classes=2, img_orig_size=448,
                 pretrained_weights_path=None):
        super().__init__()

        self.S = grid_size  # Grid size
        self.B = num_bounding_boxes  # Number of bounding boxes
        self.C = num_classes  # Number of classes
        self.img_orig_size = img_orig_size  # Original image size

        self.backbone = Darknet()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 7 * 7 * (5 * self.B + self.C)),
            nn.Sigmoid()
        )

        if pretrained_weights_path:
            self._load_pretrained_weights(pretrained_weights_path)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
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
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
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

    @torch.no_grad()
    def inference(self, images, conf_threshold=0.5, nms_threshold=0.4):
        """
        Perform inference on a batch of images.

        Args:
            images (Tensor): Batch of images, shape [N, 3, H, W]
            conf_threshold (float): Confidence threshold for filtering boxes
            nms_threshold (float): IoU threshold for non-maximum suppression

        Returns:
            list of tuples: Each tuple contains (bboxes, scores, labels) for an image
        """
        self.eval()
        outputs = self.forward(images)
        batch_size = outputs.size(0)
        results = []

        for i in range(batch_size):
            output = outputs[i]

            # [S, S, B, 4], [S, S, B], [S, S, B]
            bboxes, scores, labels = self._process_output(output)

            # 先过滤掉置信度不足的
            mask = scores > conf_threshold  # [S, S, B]
            bboxes = bboxes[mask]  # [N, 4]
            scores = scores[mask]  # [N,]
            labels = labels[mask]  # [N,]

            # apply nms
            bboxes_xyxy = xywh_to_xyxy(bboxes)
            keep = torch.zeros_like(labels, dtype=torch.bool)  # [N,]
            for c in range(self.C):
                inds = torch.where(labels == c)[0]  # [N]
                class_bboxes = bboxes_xyxy[inds]  # [CN, 4]
                class_score = scores[inds]  # [CN,]
                keep_ind = nms(class_bboxes, class_score, nms_threshold)  # [CN,]

                keep[inds[keep_ind]] = True

            bboxes = bboxes[keep]  # [N, 4]
            labels = labels[keep]  # [N,]
            scores = scores[keep]  # [N,]

            results.append((bboxes, scores, labels))

        return results

    def _process_output(self, output):
        """
        Process the raw output from the model.

        Args:
            output (Tensor): Raw output from the model, shape [S, S, B*5 + C]

        Returns:
            tuple: Processed bounding boxes, confidences, and class probabilities
        """
        S, B = self.S, self.B
        bboxes_output = output[..., :B * 5]  # [S, S, B*5]
        class_probs = output[..., B * 5:]  # [S, S, C]

        # Reshape the output tensor to separate bounding boxes and class probabilities
        bboxes_output = bboxes_output.view(S, S, B, 5)  # [S, S, B, 5]

        # Extract bounding boxes and confidences
        bboxes = bboxes_output[..., :4]
        confidences = bboxes_output[..., 4]  # [S, S, B]

        bboxes = self.decode_boxes(bboxes, self.S, self.img_orig_size)

        scores = confidences.unsqueeze(-1) * class_probs.unsqueeze(2)  # (S, S, B, C)

        # Get max scores of each bbox
        scores, labels = torch.max(scores, dim=-1)  # [S, S, B]

        return bboxes, scores, labels


if __name__ == '__main__':
    from utils.losses import YOLOv1Loss

    target = [{'boxes': torch.tensor([[0.1339, 0.4799, 0.4152, 0.2612]]), 'labels': [1]}]
    c = YOLOv1Loss().convert_to_yolo_format(target, 'cpu')
    r = YOLOv1().inference(c)
    print(r)
