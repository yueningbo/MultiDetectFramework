import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.utils import nms
from typing import List, Dict
import logging


def evaluate_model(model: torch.nn.Module, test_loader, coco_annotation_file: str,
                   device: torch.device) -> None:
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

    logging.info("Starting evaluation process.")
    for batch_idx, (images, _, img_ids) in enumerate(test_loader):
        logging.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
        images = images.to(device)
        outputs = model.inference(images)

        # Process each image in the batch
        for output, img_id in zip(outputs, img_ids):
            predictions = convert_outputs_to_coco_format(output, img_id)
            all_predictions.extend(predictions)

    if all_predictions:
        logging.info("All predictions generated. Loading ground truth annotations.")
        # Load ground truth and predictions
        coco_gt = COCO(coco_annotation_file)
        coco_dt = coco_gt.loadRes(all_predictions)

        logging.info("Running COCO evaluation.")
        # Perform evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        logging.info("Evaluation completed successfully.")
    else:
        logging.warning("No predictions were generated during evaluation.")


def convert_outputs_to_coco_format(output: List[List[float]], img_id: int) -> List[Dict]:
    """
    Convert model output to COCO format.

    Args:
        output (List[List[float]]): Model predictions for an image.
        img_id (int): Image ID.

    Returns:
        List[Dict]: Predictions in COCO format.
    """
    coco_predictions = []
    for detection in output:
        x_min, y_min, width, height, class_id, score = detection
        coco_pred = {
            "image_id": img_id,
            "category_id": int(class_id),
            "bbox": [x_min, y_min, width, height],
            "score": float(score)
        }
        coco_predictions.append(coco_pred)
    return coco_predictions


def apply_nms_per_class(predictions: List[Dict], nms_threshold: float) -> List[Dict]:
    """
    Apply Non-Maximum Suppression (NMS) per class and format the predictions.

    Parameters:
        predictions (List[Dict]): List of bounding boxes with scores.
        nms_threshold (float): IoU threshold for NMS.

    Returns:
        List[Dict]: List of predictions in COCO format after NMS.
    """
    logging.debug("Applying NMS.")
    if not predictions:
        return []

    # Group predictions by class
    by_class = {}
    for pred in predictions:
        cls_id = pred['category_id']
        if cls_id not in by_class:
            by_class[cls_id] = {'boxes': [], 'scores': [], 'ids': []}
        by_class[cls_id]['boxes'].append(pred['bbox'])
        by_class[cls_id]['scores'].append(pred['score'])
        by_class[cls_id]['ids'].append(pred['image_id'])

    final_predictions = []
    for cls_id, data in by_class.items():
        boxes = torch.tensor(data['boxes'], dtype=torch.float32)
        scores = torch.tensor(data['scores'], dtype=torch.float32)
        keep = nms(boxes, scores, nms_threshold)

        for idx in keep:
            box = boxes[idx].tolist()
            score = scores[idx].item()
            final_predictions.append({
                "image_id": data['ids'][idx],
                "category_id": cls_id,
                "bbox": box,
                "score": score
            })

    logging.debug(f"NMS produced {len(final_predictions)} predictions.")
    return final_predictions
