import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from utils.visualization import visualize_predictions


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
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, img, boxes):
        # Ensure the canvas_size is correctly initialized with img size
        boxes = BoundingBoxes(boxes, format='CXCYWH', canvas_size=img.shape[-2:])
        h, w = img.shape[-2:]
        pad_horizontal = True if h > w else False
        pad_vertical = not pad_horizontal
        pad_size = abs(h - w)

        # Apply transformations including resizing
        transforms = v2.Compose([
            v2.Pad((pad_size * pad_horizontal, pad_size * pad_vertical)),
            v2.Resize(size=self.size, antialias=True),
            v2.RandomHorizontalFlip(p=1),
        ])

        out_img, out_boxes = transforms(img, boxes)

        return out_img, out_boxes


if __name__ == '__main__':
    # 测试代码
    img_dir = 'data/datasets/open-images-bus-trucks/images'  # 替换为你的图像目录路径
    img_file = '0a5b2310fe6e429f.jpg'  # 替换为实际的图像文件名
    img = read_image(f'{img_dir}/{img_file}')
    boxes = torch.tensor([[0.5028125, 0.5046905, 0.880625, 0.990619]])  # 示例注释
    img_h, img_w = img.shape[-2:]
    boxes[:, (0, 2)] *= img_w
    boxes[:, (1, 3)] *= img_h

    # 创建转换对象
    transform = YOLOv1Transform(size=(224, 224))

    # 应用转换
    out_img, out_boxes = transform(img, boxes)

    targets = {'boxes': out_boxes, 'labels': [1]}

    visualize_predictions([out_img], [targets], [img_file])
