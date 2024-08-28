import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from matplotlib import patches

from data.transforms.transform import YOLOv1Transform


# 自定义 collate_fn 函数
def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    annotations = [torch.tensor(ann, dtype=torch.float32) for ann in annotations]
    max_num_annot = max(len(ann) for ann in annotations)
    padded_annotations = [torch.cat((ann, torch.zeros(max_num_annot - len(ann), 5))) for ann in annotations]
    annotations = torch.stack(padded_annotations, dim=0)
    return images, annotations


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
            print(image)

        return image, annotations


def test_dataset():
    img_dir = 'data/datasets/open-images-bus-trucks/images'
    annotation_dir = 'data/datasets/open-images-bus-trucks/yolo_labels/all/labels'
    transform = YOLOv1Transform()
    dataset = BusAndTruckDataset(img_dir, annotation_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for i, (images, annotations) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  Image batch size: {images.size()}")  # 打印图像的批次大小
        print(f"  Annotation batch size: {annotations.size()}")  # 打印标注的批次大小

        for j in range(images.size(0)):
            single_image = images[j].permute(1, 2, 0)
            single_image = single_image.numpy()
            single_image = Image.fromarray(single_image.astype('uint8'))

            fig, ax = plt.subplots(1)
            ax.imshow(single_image)
            ax.set_title(f"Image {j + 1}")

            for ann in annotations[j]:
                class_id, x_center, y_center, width, height = ann
                x_min = int((x_center - width / 2) * single_image.width)
                y_min = int((y_center - height / 2) * single_image.height)
                x_max = int((x_center + width / 2) * single_image.width)
                y_max = int((y_center + height / 2) * single_image.height)
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)

            plt.show()


if __name__ == '__main__':
    test_dataset()
