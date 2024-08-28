import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_predictions(images, annotations, img_file):
    print(images.shape)  # example:  torch.Size([2, 3, 224, 224])
    print(
        annotations)  # example:  [tensor([[  0.0000, 106.6800, 133.8400,  97.1600, 105.6532]]), tensor([[  1.0000, 107.2400, 126.3973, 190.9600, 151.4525]])]
    print(img_file)  # example: 'a32b89b2580a6e81.jpg', 'ec10f38886481c7a.jpg']
    """
    
['a32b89b2580a6e81.jpg', 'ec10f38886481c7a.jpg']
    Visualize the predictions on the images.
    """
    for i, (image, annotation) in enumerate(zip(images, annotations)):
        image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        image = image.numpy()  # 转换为 numpy 数组
        plt.imshow(image)
        ax = plt.gca()

        for ann in annotation:
            class_id, x_center, y_center, width, height = ann
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        plt.show()
