import json
import os
import torch
import logging
from data.loaders.dataset_loader import get_loader
from utils.losses import YoloV1Loss
from utils.metrics import evaluate_model
from models.yolov1.yolov1_model import YOLOv1
from utils.utils import print_model_flops

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:
    def __init__(self, config_path, weights_path, amp):
        self.config = self.load_config(config_path)
        self.weights_path = weights_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLOv1(self.config['grid_size'], self.config['num_bounding_boxes'], self.config['num_classes']).to(
            self.device)
        print_model_flops(self.model, (3, 448, 448), self.device)
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                         momentum=self.config['momentum'], weight_decay=self.config['decay'])
        self.train_loader, self.test_loader = get_loader(self.config, self.device)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for images, targets, img_file in self.train_loader:
            images = images.to(self.device)

            self.optimizer.zero_grad()

            # 自动选择数据类型
            with torch.cuda.amp.autocast(enabled=bool(self.scaler)):
                outputs = self.model(images)
                criterion = YoloV1Loss(device=self.device)
                loss = criterion(outputs, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        return evaluate_model(self.model, self.test_loader, self.config['val_annotation_path'], self.device)

    def save_model_weights(self, epoch):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save(self.model.state_dict(), f'{self.weights_path}_epoch_{epoch}.pth')

    def train(self):
        for epoch in range(self.config['epochs']):
            loss = self.train_epoch()
            logging.info(f'Epoch {epoch + 1}/{self.config["epochs"]}, Loss: {loss}')
            self.evaluate()

            if (epoch + 1) % 10 == 0:
                self.save_model_weights(epoch + 1)

        self.save_model_weights('final')


if __name__ == "__main__":
    config_path = 'configs/yolov1.json'
    weights_path = 'outputs/yolov1/model_weights.pth'
    amp = True
    trainer = Trainer(config_path, weights_path, amp)
    trainer.train()
