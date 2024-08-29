import json
import os
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from data.loaders.dataset_loader import get_loader
from utils.losses import YoloV1Loss
from utils.metrics import evaluate_model
from models.yolov1.yolov1_model import YOLOv1
from utils.utils import print_model_flops

from utils.warmup_scheduler import WarmUpScheduler

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
writer = SummaryWriter(log_dir='outputs/yolov1')


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

        # Initialize the WarmUpScheduler
        self.scheduler = WarmUpScheduler(self.optimizer, warmup_epochs=self.config['warmup_epochs'],
                                         max_lr=self.config['learning_rate'])

        self.train_loader, self.val_loader = get_loader(self.config, self.device)
        logging.info(f'Configuration loaded from {config_path}')
        logging.info(f'Weights will be saved to {weights_path}')
        logging.info(f'Using device: {self.device}')

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f'Configuration: {config}')
        return config

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        logging.info(f'Starting epoch {epoch + 1}')

        for batch_idx, (images, targets, img_id) in enumerate(self.train_loader):
            images = images.to(self.device)

            self.optimizer.zero_grad()

            # Automatic mixed precision training
            with torch.cuda.amp.autocast(enabled=bool(self.scaler)):
                outputs = self.model(images)
                criterion = YoloV1Loss(device=self.device)
                loss = criterion(outputs, targets)

            # Record loss
            writer.add_scalar('Loss/train', loss.item(), global_step=epoch)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)  # Unscaling needed for AMP
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                self.optimizer.step()

            total_loss += loss.item()

            # Log details of the current batch
            if batch_idx % 10 == 0:  # Log every 10 batches
                logging.info(f'Epoch {epoch + 1}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item()}')

        avg_loss = total_loss / len(self.train_loader)
        logging.info(f'Epoch {epoch + 1} completed, Average Loss: {avg_loss}')
        return avg_loss

    def evaluate(self):
        evaluate_model(self.model, self.val_loader, self.config['val_annotation_path'], self.device)

    def save_model_weights(self, epoch):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        weights_file = f'{self.weights_path}_epoch_{epoch}.pth'
        torch.save(self.model.state_dict(), weights_file)
        logging.info(f'Model weights saved to {weights_file}')

    def train(self):
        for epoch in range(self.config['epochs']):
            self.train_epoch(epoch)

            if (epoch + 1) % 10 == 0:
                self.evaluate()

            if (epoch + 1) % 10 == 0:
                self.save_model_weights(epoch + 1)

            # Update the learning rate
            self.scheduler.step()

        self.save_model_weights('final')
        logging.info('Training completed')


if __name__ == "__main__":
    config_path = 'configs/yolov1.json'
    weights_path = 'outputs/yolov1/model_weights.pth'
    amp = True
    trainer = Trainer(config_path, weights_path, amp)
    trainer.train()
    writer.close()
