import json

import torch
from data.loaders.dataset_loader import load_data
from utils.losses import compute_loss
from utils.metrics import evaluate_model
from models.yolov1.yolov1_model import YOLOv1


def main(config_path, weights_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model
    model = YOLOv1(config)

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                                weight_decay=config['decay'])

    # Load data
    train_loader, test_loader = load_data(config)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{config.epochs}, Loss: {loss.item()}')

        # Evaluation
        evaluate_model(model, test_loader)

        # Save weights
        if (epoch + 1) % 10 == 0:
            model.save_weights(f'{weights_path}_epoch_{epoch + 1}.pth')

    model.save_weights(weights_path)


if __name__ == "__main__":
    config_path = 'configs/yolov1.json'
    weights_path = 'outputs/yolov1/model_weights.pth'
    main(config_path, weights_path)
