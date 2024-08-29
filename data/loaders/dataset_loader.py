from torch.utils.data import DataLoader

from data.loaders.coco_dataset import COCODataset, collate_fn
from data.transforms.transform import YOLOv1Transform


def get_loader(config, device):
    # 创建数据转换
    transform = YOLOv1Transform()

    # 加载train数据集
    train_dataset = COCODataset(img_dir=config['img_dir'],
                                annotation_path=config['train_annotation_path'],
                                transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, device))

    # 加载val数据集
    val_dataset = COCODataset(img_dir=config['img_dir'],
                              annotation_path=config['val_annotation_path'],
                              transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=lambda batch: collate_fn(batch, device))

    return train_loader, val_loader
