import json

import torch
from models.yolov1.yolov1_model import YOLOv1
from data.loaders.dataset_loader import load_data
from utils.visualization import visualize_predictions


def main():
    model_config = json.loads('configs/yolov1.json')
    model = YOLOv1(model_config)
    model.load_weights('path/to/load/weights')
    test_loader = load_data(config)

    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images)
            visualize_predictions(images, outputs)


if __name__ == "__main__":
    main()
