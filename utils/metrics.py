import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging


def evaluate_model(model: torch.nn.Module, test_loader, coco_annotation_file: str, device: torch.device) -> None:
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


def convert_outputs_to_coco_format(output, img_id):
    """
    Convert model output to COCO format for evaluation.

    Parameters:
        output (tuple): Output from model inference containing (bboxes, scores, labels).
        img_id (int): Image ID corresponding to the output.

    Returns:
        list[dict]: List of predictions in COCO format.
    """
    bboxes, scores, labels = output
    coco_predictions = []
    for bbox, score, label in zip(bboxes, scores, labels):
        x_min, y_min, width, height = bbox.tolist()
        coco_prediction = {
            "image_id": img_id,
            "category_id": label.item() + 1,
            "bbox": [x_min, y_min, width, height],
            "score": score.item()
        }
        coco_predictions.append(coco_prediction)

    return coco_predictions
