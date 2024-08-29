import json
import torch
from models.yolov1.yolov1_model import YOLOv1
from data.loaders.dataset_loader import get_loader
from utils.metrics import evaluate_model


def main():
    config_path = 'configs/yolov1.json'
    weights_path = 'outputs/yolov1/model_weights.pth'
    coco_annotation_file = 'data/annotations/instances_val2017.json'  # Path to COCO annotations

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLOv1(config['grid_size'], config['num_bounding_boxes'], config['num_classes']).to(device)
    model.load_state_dict(torch.load(weights_path))

    # Load test data
    _, test_loader = get_loader(config, device)

    # Evaluate the model
    evaluate_model(model, test_loader, coco_annotation_file)


if __name__ == "__main__":
    main()
