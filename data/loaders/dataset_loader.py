import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import json
from data.transforms.transform import YOLOv1Transform
from utils.visualization import visualize_predictions
from pycocotools.coco import COCO
from tqdm import tqdm


# 自定义 collate_fn 函数
def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    filenames = [item[2] for item in batch]  # 收集文件名
    images = torch.stack(images, dim=0)
    return images, annotations, filenames  # 返回图像、注释和文件名


class BusAndTruckDataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None):
        self.img_dir = img_dir
        self.annotation_path = annotation_path
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.annotations = self.load_coco_annotation(annotation_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, file_name)
        image = read_image(img_path)

        # 直接从索引中获取注释
        annotation = self.annotations.get(file_name, None)
        if annotation is None:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int8)
        else:
            boxes = torch.tensor(annotation['boxes'], dtype=torch.float32)
            labels = torch.tensor(annotation['labels'], dtype=torch.int8)

        # 把比例转成绝对值
        img_h, img_w = image.shape[-2:]
        boxes[:, (0, 2)] *= img_w
        boxes[:, (1, 3)] *= img_h

        if self.transform:
            image, boxes = self.transform(image, boxes)

        targets = {
            'boxes': boxes,
            'labels': labels
        }

        return image, targets, file_name

    def load_coco_annotation(self, annotation_path):
        annotations = {}
        with open(annotation_path, 'r') as file:
            coco_data = json.load(file)
        for ann in tqdm(coco_data['annotations'], desc='Indexing annotations'):
            img_id = ann['image_id']
            file_name = coco_data['images'][img_id]['file_name']
            if file_name not in annotations:
                annotations[file_name] = {'boxes': [], 'labels': []}
            x, y, width, height = ann['bbox']
            annotations[file_name]['boxes'].append([x, y, width, height])
            annotations[file_name]['labels'].append(ann['category_id'])

        return annotations


def test_dataset():
    img_dir = 'data/datasets/open-images-bus-trucks/images'
    annotation_dir = r'data\datasets\open-images-bus-trucks\annotations\micro_open_images_train_coco_format.json'
    transform = YOLOv1Transform()
    dataset = BusAndTruckDataset(img_dir, annotation_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for i, (images, annotations, img_file) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  Image batch size: {len(images)}")  # 打印图像的批次大小
        print(f"  Annotation batch size: {len(annotations)}")  # 打印标注的批次大小

        visualize_predictions(images, annotations, img_file)


if __name__ == '__main__':
    test_dataset()
