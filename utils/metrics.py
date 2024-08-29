import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms
from typing import List, Dict


def evaluate_model(model, test_loader, coco_annotation_file, device) -> None:
    """
    Evaluate the model using the COCO dataset and compute metrics.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        coco_annotation_file (str): Path to COCO annotation file.
        device (torch.device): The device to run the model on.
    """
    model.eval()
    all_predictions = []

    for images, targets, img_ids in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)

        # Process each image in the batch
        for output, img_id in zip(outputs, img_ids):
            predictions = convert_outputs_to_coco_format(output, img_id)
            all_predictions.extend(predictions)

    if all_predictions:
        # Load ground truth and predictions
        coco_gt = COCO(coco_annotation_file)
        coco_dt = coco_gt.loadRes(all_predictions)

        # Perform evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def convert_outputs_to_coco_format(output: torch.Tensor, img_id: int, threshold: float = 0.5,
                                   nms_threshold: float = 0.4) -> List[Dict]:
    """
    Convert YOLO output tensors to COCO format with NMS applied.

    Parameters:
        output (torch.Tensor): YOLO model output with shape (S, S, C + B * 5).
        img_id (int): Image ID.
        threshold (float): Confidence score threshold.
        nms_threshold (float): IoU threshold for NMS.

    Returns:
        List[Dict]: List of predictions in COCO format.
    """
    predictions = []
    B = 2  # Number of bounding boxes
    grid_size = output.shape[1]
    output = output.permute(1, 2, 0).contiguous().view(grid_size, grid_size, -1)

    for i in range(grid_size):
        for j in range(grid_size):
            grid_cell = output[i, j, :]

            # Collect boxes, scores, and classes
            boxes, scores, classes = [], [], []

            for b in range(B):
                box = grid_cell[b * 5:(b + 1) * 5]
                confidence = box[4].item()

                if confidence > threshold:
                    x_center, y_center, w, h = box[:4]
                    class_scores = grid_cell[B * 5:]
                    class_id = torch.argmax(class_scores).item()
                    class_score = class_scores[class_id].item()

                    # Compute absolute bounding box coordinates
                    x_min = (x_center + j / grid_size - w / 2) * 448
                    y_min = (y_center + i / grid_size - h / 2) * 448
                    width = w * 448
                    height = h * 448

                    boxes.append([x_min, y_min, width, height])
                    scores.append(confidence * class_score)
                    classes.append(class_id)

            # Apply NMS per class
            predictions.extend(apply_nms_per_class(boxes, scores, classes, img_id, nms_threshold))

    return predictions


def apply_nms_per_class(boxes: List[List[float]], scores: List[float], classes: List[int], img_id: int,
                        nms_threshold: float) -> List[Dict]:
    """
    Apply Non-Maximum Suppression (NMS) per class and format the predictions.

    Parameters:
        boxes (List[List[float]]): List of bounding boxes.
        scores (List[float]): List of confidence scores.
        classes (List[int]): List of class IDs.
        img_id (int): Image ID.
        nms_threshold (float): IoU threshold for NMS.

    Returns:
        List[Dict]: List of predictions in COCO format after NMS.
    """
    predictions = []
    unique_classes = set(classes)

    for cls_id in unique_classes:
        cls_boxes = [box for box, cls in zip(boxes, classes) if cls == cls_id]
        cls_scores = [score for score, cls in zip(scores, classes) if cls == cls_id]

        if cls_boxes:
            cls_boxes = torch.tensor(cls_boxes, dtype=torch.float32)
            cls_scores = torch.tensor(cls_scores, dtype=torch.float32)
            keep = nms(cls_boxes, cls_scores, nms_threshold)

            for idx in keep:
                box = cls_boxes[idx].tolist()
                score = cls_scores[idx].item()
                predictions.append({
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": box,
                    "score": score
                })

    return predictions
