from torch.utils.data import DataLoader

from data.loaders.coco_dataset import COCODataset, collate_fn
from data.transforms.transform import YOLOv1Transform


def get_loader(config):
    # 创建数据转换
    transform = YOLOv1Transform()

    # 加载训练数据集
    train_dataset = COCODataset(img_dir=config['img_dir'],
                                annotation_path=config['train_annotation_path'],
                                transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    # 加载测试数据集
    test_dataset = COCODataset(img_dir=config['img_dir'],
                               annotation_path=config['test_annotation_path'],
                               transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader
