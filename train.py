import json
import os
import torch
import logging
from data.loaders import get_loader
from utils.losses import compute_loss
from utils.metrics import evaluate_model
from models.yolov1.yolov1_model import YOLOv1
from utils.utils import print_model_flops

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def train_epoch(model, optimizer, train_loader, scaler, device):
    model.train()
    total_loss = 0
    for images, targets, img_file in train_loader:
        images = images.to(device)
        targets = targets

        optimizer.zero_grad()

        # 自动选择数据类型
        with torch.cuda.amp.autocast(enabled=bool(scaler)):
            outputs = model(images)
            loss = compute_loss(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model_wrapper(model, test_loader, device):
    return evaluate_model(model, test_loader)


def save_model_weights(model, weights_path, epoch):
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), f'{weights_path}_epoch_{epoch}.pth')


def main(config_path, weights_path, amp):
    config = load_config(config_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLOv1(config['grid_size'], config['num_bounding_boxes'], config['num_classes'])
    print_model_flops(model, (3, 448, 448))
    model.to(device)

    scaler = torch.cuda.amp.GradScaler() if amp else None

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                                weight_decay=config['decay'])

    train_loader, test_loader = get_loader(config, device)

    for epoch in range(config['epochs']):
        loss = train_epoch(model, optimizer, train_loader, scaler, device)
        logging.info(f'Epoch {epoch + 1}/{config["epochs"]}, Loss: {loss}')

        evaluate_model_wrapper(model, test_loader, device)

        if (epoch + 1) % 10 == 0:
            save_model_weights(model, weights_path, epoch + 1)

    save_model_weights(model, weights_path, 'final')


if __name__ == "__main__":
    config_path = 'configs/yolov1.json'
    weights_path = 'outputs/yolov1/model_weights.pth'
    amp = True
    main(config_path, weights_path, amp)
