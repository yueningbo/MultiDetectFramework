import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from utils.visualization import visualize_prediction


class PaddingToSquare:
    def __call__(self, img, *args):
        if args:
            boxes = args[0]

        C, H, W = img.shape
        max_dim = max(H, W)
        pad_height = max_dim - H if H < max_dim else 0
        pad_width = max_dim - W if W < max_dim else 0

        return v2.Pad((0, pad_width, 0, pad_height))


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
            v2.Pad((pad_size * pad_horizontal, pad_size * pad_vertical)),
            v2.Resize(size=self.size, antialias=True),
            v2.RandomHorizontalFlip(p=1),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        out_img, out_boxes = transforms(img, boxes)

        return out_img, out_boxes


if __name__ == '__main__':
    # 测试代码
    img_dir = 'data/datasets/VOCdevkit/VOC2007/JPEGImages'  # 替换为你的图像目录路径
    img_file = '000016.jpg'  # 替换为实际的图像文件名
    img = read_image(f'{img_dir}/{img_file}').to(torch.float32)
    boxes = torch.tensor([[16, 1, 225, 170]])  # 示例注释

    # 创建转换对象
    transform = YOLOv1Transform(size=(448, 448))

    # 应用转换
    out_img, out_boxes = transform(img, boxes)

    target = {'boxes': out_boxes, 'labels': [1]}

    visualize_prediction(out_img, target, img_file)
