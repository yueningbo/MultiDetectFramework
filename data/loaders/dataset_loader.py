import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from data.transforms.transform import YOLOv1Transform
from utils.visualization import visualize_predictions
from pycocotools.coco import COCO


# 自定义 collate_fn 函数
def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    filenames = [item[2] for item in batch]  # 收集文件名
    images = torch.stack(images, dim=0)
    return images, annotations, filenames  # 返回图像、注释和文件名


class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None):
        self.img_dir = img_dir
        self.annotation_path = annotation_path
        self.transform = transform
        self.coco = COCO(annotation_path)
        self.img_files = [img['file_name'] for img in self.coco.loadImgs(self.coco.getImgIds())]
        self.img_ids = self.coco.getImgIds()
        self.cats = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, file_name)
        image = read_image(img_path)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        coco_annotation = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in coco_annotation:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
        print('boxes:', boxes)

        boxes = torch.tensor(boxes, dtype=torch.float32).view(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        targets = {
            'boxes': boxes,
            'labels': labels
        }

        return image, targets, file_name


def test_dataset():
    img_dir = 'data/datasets/VOCdevkit/VOC2007/JPEGImages'
    annotation_dir = r'data/datasets/VOCdevkit/VOC2007/test.json'
    transform = YOLOv1Transform()
    dataset = COCODataset(img_dir, annotation_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for i, (images, annotations, img_file) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  Image batch size: {len(images)}")  # 打印图像的批次大小
        print(f"  Annotation batch size: {len(annotations)}")  # 打印标注的批次大小

        visualize_predictions(images, annotations, img_file)


if __name__ == '__main__':
    test_dataset()
