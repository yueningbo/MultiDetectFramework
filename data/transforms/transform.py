import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from utils.visualization import visualize_prediction


class YOLOv1Transform:
    def __init__(self, size=(448, 448)):
        self.size = size

    def __call__(self, img, boxes):
        # Ensure the canvas_size is correctly initialized with img size
        boxes = BoundingBoxes(boxes, format='XYWH', canvas_size=img.shape[-2:])
        h, w = img.shape[-2:]
        pad_horizontal = True if h > w else False
        pad_vertical = not pad_horizontal
        pad_size = abs(h - w)

        # Apply transformations including resizing
        transforms = v2.Compose([
            # v2.Pad((pad_size * pad_horizontal, pad_size * pad_vertical)),
            # v2.Resize(size=self.size, antialias=True),
            # v2.RandomHorizontalFlip(p=1),
            # 图片归一化
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        out_img, out_boxes = transforms(img, boxes)

        # 图像的尺寸
        canvas_size = out_boxes.canvas_size

        # 归一化边界框
        normalized_boxes = out_boxes / torch.tensor([canvas_size[0], canvas_size[1], canvas_size[0], canvas_size[1]])

        return out_img, normalized_boxes


if __name__ == '__main__':
    # 测试代码
    # img_dir = 'data/datasets/VOCdevkit/VOC2007/JPEGImages'  # 替换为你的图像目录路径
    img_dir = 'data/datasets/Mnist/mnist_train'  # 替换为你的图像目录路径
    img_file = '000001.jpg'  # 替换为实际的图像文件名
    img = read_image(f'{img_dir}/{img_file}').to(torch.float32) / 255
    boxes = torch.tensor([[302, 332, 56, 56], [127, 137, 56, 56]])  # 示例注释

    # 创建转换对象
    transform = YOLOv1Transform(size=(448, 448))

    # 应用转换
    out_img, out_boxes = transform(img, boxes)

    target = {'boxes': out_boxes, 'labels': [1, 2]}

    visualize_prediction(out_img, target, img_file)
