def evaluate_model(outputs, targets):
    """
    Evaluate the model performance.
    """

    # Calculate precision and recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for output, target in zip(outputs, targets):
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, 5:]

        target_boxes = target[:, :4]
        target_labels = target[:, 5:]

        highest_pred_scores, _ = pred_scores.max(dim=1)
        highest_pred_labels = pred_labels.argmax(dim=1)

        IoU = bbox_iou(pred_boxes, target_boxes)

        for i in range(len(IoU)):
            if IoU[i].item() > 0.5 and highest_pred_labels[i] == target_labels[i]:
                true_positives += 1
            else:
                false_positives += 1

        false_negatives = len(target_labels) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
