import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from data.transforms.transform import YOLOv1Transform
from utils.visualization import visualize_predictions


# 自定义 collate_fn 函数
def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    filenames = [item[2] for item in batch]  # 收集文件名
    images = torch.stack(images, dim=0)
    return images, annotations, filenames  # 返回图像、注释和文件名


class BusAndTruckDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')

        annotation_path = os.path.join(self.annotation_dir,
                                       self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        annotations = []
        with open(annotation_path, 'r') as file:
            for line in file.readlines():
                class_id, x_center, y_center, width, height = map(float, line.split())
                annotations.append([class_id, x_center, y_center, width, height])

        annotations = torch.tensor(annotations, dtype=torch.float32).view(-1, 5)

        if self.transform:
            image, annotations = self.transform(image, annotations)

        return image, annotations, self.img_files[idx]  # 返回图像、注释和文件名


def test_dataset():
    img_dir = 'data/datasets/open-images-bus-trucks/images'
    annotation_dir = 'data/datasets/open-images-bus-trucks/yolo_labels/all/labels'
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
