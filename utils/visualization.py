import matplotlib.pyplot as plt
from matplotlib import patches


def visualize_predictions(images, targets, img_files):
    for i, (image, target, img_file) in enumerate(zip(images, targets, img_files)):
        print(f"Visualizing predictions for image: {img_file}")
        image = image.permute(1, 2, 0)  # 从 (C, H, W) 转换为 (H, W, C)
        image = image.numpy()  # 转换为 numpy 数组
        h, w = image.shape[:2]
        plt.imshow(image)
        ax = plt.gca()

        boxes = target['boxes']
        labels = target['labels']

        for i, b in enumerate(boxes):
            x_center, y_center, width, height = b.tolist()
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'Class: {labels[i]}', bbox=dict(facecolor='white', alpha=0.5))

        plt.title(f"Image {i + 1}: {img_file}")
        plt.show()
