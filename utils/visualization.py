import matplotlib.pyplot as plt
import torch
from matplotlib import patches


def visualize_prediction(image, target, img_file):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = image * std[:, None, None] + mean[:, None, None]
    image = image.to(torch.int32)

    print(f"Visualizing predictions for image: {img_file}")
    image = image.permute(1, 2, 0)  # 从 (C, H, W) 转换为 (H, W, C)
    image = image.numpy()  # 转换为 numpy 数组
    plt.imshow(image)
    ax = plt.gca()

    boxes = target['boxes']
    labels = target['labels']

    for i, b in enumerate(boxes):
        x_center, y_center, width, height = b.tolist()
        x_min = x_center
        y_min = y_center
        x_max = x_center + width
        y_max = y_center + height
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, f'Class: {labels[i]}', bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"Image: {img_file}")
    plt.show()


def visualize_predictions(images: torch.Tensor, targets: list, img_files: list):
    for i, (image, target, img_file) in enumerate(zip(images, targets, img_files)):
        visualize_prediction(image, target, img_file)
