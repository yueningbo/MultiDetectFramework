import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms


def evaluate_model(model, test_loader, coco_annotation_file, device):
    model.eval()
    all_predictions = []

    for images, targets, img_ids in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)

        # Process each image in the batch
        for output, img_id in zip(outputs, img_ids):
            # Convert outputs to COCO format
            predictions = convert_outputs_to_coco_format(output, img_id)
            all_predictions.extend(predictions)

    # Load ground truth and predictions
    coco_gt = COCO(coco_annotation_file)

    coco_dt = coco_gt.loadRes(all_predictions)

    # Perform evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def convert_outputs_to_coco_format(output, img_id, threshold=0.5, nms_threshold=0.4):
    """
    Convert YOLO output tensors to COCO format with NMS applied.

    Args:
        output (Tensor): YOLO model output with shape (S, S, C + B * 5).
        img_id (int): image_id.
        threshold (float): Confidence score threshold to filter out low confidence predictions.
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
            boxes = []
            scores = []
            classes = []

            for b in range(B):
                box = grid_cell[b * 5:(b + 1) * 5]
                confidence = box[4].item()

                if confidence > threshold:
                    x_center = (box[0] + j) / grid_size
                    y_center = (box[1] + i) / grid_size
                    w = box[2]
                    h = box[3]

                    # Get class scores
                    class_scores = grid_cell[B * 5:]
                    class_id = torch.argmax(class_scores).item()
                    class_score = class_scores[class_id].item()

                    # Compute absolute bounding box coordinates
                    x_min = (x_center - w / 2) * 448
                    y_min = (y_center - h / 2) * 448
                    width = w * 448
                    height = h * 448

                    boxes.append([x_min, y_min, width, height])
                    scores.append(confidence * class_score)
                    classes.append(class_id)

            # Apply NMS per class
            for cls_id in set(classes):
                cls_boxes = [box for box, cls in zip(boxes, classes) if cls == cls_id]
                cls_scores = [score for score, cls in zip(scores, classes) if cls == cls_id]

                if len(cls_boxes) > 0:
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
