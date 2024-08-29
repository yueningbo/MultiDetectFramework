import json
import os
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from data.loaders.dataset_loader import get_loader
from utils.losses import YoloV1Loss
from utils.metrics import evaluate_model
from models.yolov1.yolov1_model import YOLOv1
from utils.utils import print_model_flops, load_config

from utils.warmup_scheduler import WarmUpScheduler

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:
    def __init__(self,
                 config_path,
                 weights_path,
                 amp=False,
                 pretrained_weights_path=None,
                 freeze_backbone_epoch=20,
                 summary_writer_path=None):
        self.config = load_config(config_path)
        self.weights_path = weights_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = YOLOv1(self.config['grid_size'],
                            self.config['num_bounding_boxes'],
                            self.config['num_classes'],
                            pretrained_weights_path=pretrained_weights_path
                            ).to(self.device)

        print_model_flops(self.model, (3, 448, 448), self.device)
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                         momentum=self.config['momentum'], weight_decay=self.config['decay'])

        # Initialize the WarmUpScheduler
        self.scheduler = WarmUpScheduler(self.optimizer, warmup_epochs=self.config['warmup_epochs'],
                                         max_lr=self.config['learning_rate'])

        self.train_loader, self.val_loader = get_loader(self.config, self.device)
        self.freeze_backbone_epoch = freeze_backbone_epoch
        self.writer = SummaryWriter(log_dir=summary_writer_path)

        logging.info(f'Configuration loaded from {config_path}')
        logging.info(f'Weights will be saved to {weights_path}')
        logging.info(f'Using device: {self.device}')

    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        logging.info("Backbone parameters frozen.")

    def unfreeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        logging.info("Backbone parameters unfrozen.")

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
                criterion = YoloV1Loss(self.config, device=self.device)
                loss = criterion(outputs, targets)

            # Record loss
            self.writer.add_scalar('Loss/train', loss.item(), global_step=epoch)

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
        # Freeze backbone parameters
        self.freeze_backbone()
        for epoch in range(self.config['epochs']):
            if epoch == self.freeze_backbone_epoch:
                self.unfreeze_backbone()

            self.train_epoch(epoch)

            if (epoch + 1) % 5 == 0:
                self.evaluate()

            if (epoch + 1) % 10 == 0:
                self.save_model_weights(epoch + 1)

            # Update the learning rate
            self.scheduler.step()

        self.save_model_weights('final')
        self.writer.close()
        logging.info('Training completed')


if __name__ == "__main__":
    os.chdir('../')
    config_path = 'configs/yolov1.json'
    weights_path = 'outputs/yolov1/model_weights.pth'
    amp = True
    summary_writer_path = 'outputs/yolov1'
    pretrained_weights_path = 'outputs/yolov1/model_weights.pth_epoch_20.pth'
    freeze_backbone_epoch = 20

    trainer = Trainer(
        config_path, weights_path, amp,
        summary_writer_path=summary_writer_path,
        # pretrained_weights_path=pretrained_weights_path,
        freeze_backbone_epoch=freeze_backbone_epoch
    )

    trainer.train()
